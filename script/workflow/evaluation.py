"""
Evaluate subagent research on HotpotQA or BrowseComp questions.

Datasets:
- HotpotQA: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- BrowseComp: local/data/BrowseCompPlus/data/browsecomp_qa_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Callable

from datasets import load_dataset
from transformers import AutoTokenizer
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table

from camel.toolkits import FunctionTool, SearchToolkit

from rosetta.workflow.retriever import search_engine
from rosetta.workflow.browse_searcher import search, get_document, configure_search
from rosetta.workflow.selector import ContextSelector
from rosetta.workflow.track import InteractionTracker, TreeTracker
from rosetta.workflow.evaluation import (
    extract_answer,
    exact_match,
    load_done_ids,
    run_research,
    LLMJudge,
)
from rosetta.workflow.camel_utils import create_model, setup_env
from rosetta.workflow.basic_utils import ContentMode, HistoryConfig


@dataclass
class EvalRecord:
    idx: int
    example_id: str
    question: str
    gold_answer: str
    pred_answer: str
    pred_raw: str
    llm0_messages: Optional[list[dict[str, Any]]]
    correct_em: bool
    seconds: float
    rounds: Optional[int] = None
    tools_used: Optional[list[list[str]]] = None
    state_sequence: Optional[list[str]] = None
    error: Optional[str] = None
    usage: Optional[dict[str, Any]] = None


@dataclass
class EvalConfig:
    dataset: str  # Dataset: hotpotqa, browsecomp
    data_path: Optional[str]  # Path to data file (for browsecomp)
    subset: str
    split: str
    limit: int
    max_rounds: int
    output: Path
    output_format: str
    resume: bool
    model_url: str
    model_type: Optional[str]  # None uses provider defaults
    model_provider: str  # Model provider: local, openai, gemini
    tokenizer: str
    mode: str
    main_to_search: str
    search: str
    search_to_main: str
    main: str
    tools: list[str]
    num_workers: int
    state_rule: list[str]
    judge_model_provider: str = "local"  # Model provider for LLM judge (default: local)
    judge_api_url: Optional[str] = None  # API URL for LLM judge (defaults to model_url)
    judge_model_type: Optional[str] = None  # Model type for LLM judge
    step_timeout: Optional[float] = None  # Timeout in seconds for agent step calls
    enable_thinking: bool = False  # Enable thinking mode for main agent (local models only)
    stream: bool = False  # Enable streaming responses (adds {"stream": True} to model_config_dict)
    patch: bool = False  # Patch mode: re-run only failed examples from existing output
    max_tokens: int = 32768  # Maximum tokens to generate per model call
    history_assistant: str = "full"  # History config for assistant messages: full, none, summarized
    history_tool: str = "full"  # History config for tool messages: full, none, summarized
    history_reasoning: str = "full"  # History config for reasoning messages: full, none, summarized
    history_delay: int = 0  # Delay in rounds before applying history transformations
    temperature: float = 0.0  # Sampling temperature for model generation

def create_context_plan(config: EvalConfig, use_single: bool, use_tree: bool) -> Optional[dict]:
    """Create context plan based on configuration."""
    main_to_search_select_fn: Callable = {
        "all": ContextSelector.select_skip_system,
        "initial": ContextSelector.select_initial_with_system,
        "none": ContextSelector.select_none,
    }[config.main_to_search]

    search_contextual_select_fn: Callable = {
        "all": ContextSelector.select_all,
        "initial": ContextSelector.select_initial,
        "none": ContextSelector.select_none,
    }[config.search]

    search_to_main_select_fn: Callable = {
        "all": ContextSelector.select_skip_system,
        "qr": ContextSelector.select_query_response_with_system,
    }[config.search_to_main]

    main_contextual_select_fn: Callable = {
        "all": ContextSelector.select_all,
        "qr": ContextSelector.select_query_response,
    }[config.main]

    if use_single:
        return {
            "main_contextual": ContextSelector(
                select_fn=ContextSelector.select_query_response
            ),
        }
    elif not use_tree:
        return {
            'search_to_main_selector': ContextSelector(
                filter_fn=ContextSelector.filter_search_only,
                select_fn=search_to_main_select_fn
            ),
            'main_to_search_selector': ContextSelector(
                filter_fn=None,
                select_fn=main_to_search_select_fn
            ),
            'search_contextual': ContextSelector(
                select_fn=search_contextual_select_fn
            ),
            'main_contextual': ContextSelector(
                select_fn=main_contextual_select_fn
            ),
        }
    else:
        return None


def evaluate_single(
    idx: int,
    ex: dict,
    config: EvalConfig,
    model,
    search_model,
    think_model,
    tokenizer,
    tools: list[FunctionTool],
) -> EvalRecord:
    """Evaluate a single example."""
    example_id = str(ex["id"])
    question = ex["question"]
    gold = ex["answer"]

    use_single = config.mode == "single"
    use_singletool = config.mode == "singletool"
    use_tree = config.mode in ("tree", "tool")

    context_plan = create_context_plan(config, use_single or use_singletool, use_tree)

    t0 = time.time()
    pred_raw = ""
    pred = ""
    llm0_messages: Optional[list[dict[str, Any]]] = None
    state_sequence: Optional[list[str]] = None
    err: Optional[str] = None
    tracker: Optional[InteractionTracker] = None
    tree_tracker: Optional[TreeTracker] = None

    # Retry logic for rate limit errors (429)
    max_retries = 1
    for attempt in range(max_retries + 1):
        tracker = InteractionTracker(tokenizer=tokenizer)
        tree_tracker = TreeTracker() if use_tree else None

        try:
            pred_raw, tracker = run_research(
                mode=config.mode,
                question=question,
                main_model=model,
                search_model=search_model if not (use_single or use_singletool) else None,
                think_model=think_model if not (use_single or use_singletool) else None,
                tracker=tracker,
                search_tools=tools if not (use_single or use_singletool) else None,
                context_plan=context_plan,
                show_status=False,
                max_rounds=config.max_rounds,
                worker_model=search_model if use_tree else None,
                rewind_model=search_model if use_tree else None,
                exam_model=search_model if use_tree else None,
                worker_tools=tools if use_tree else None,
                tree_tracker=tree_tracker,
                state_rule_actions=config.state_rule if use_tree else None,
                main_agent_tools=tools if (use_single or use_singletool) else None,
                step_timeout=config.step_timeout,
                tokenizer=tokenizer,
                history_assistant=config.history_assistant if use_singletool else None,
                history_tool=config.history_tool if use_singletool else None,
                history_reasoning=config.history_reasoning if use_singletool else None,
                history_delay=config.history_delay if use_singletool else None,
            )
            # Success - process results
            if use_tree and tracker is not None:
                state_sequence = tracker.state_sequence
            extracted = extract_answer(pred_raw)
            pred = extracted if extracted is not None else pred_raw.strip()
            try:
                llm0_messages = tracker.get_messages(llm_id=0) if tracker is not None else None
            except Exception:
                llm0_messages = None
            break  # Success, exit retry loop
        except Exception as e:
            # Check if rate limit error (429) and should retry
            is_rate_limit = (
                "RateLimitError" in type(e).__name__
                or ("429" in str(e) and "too_many_requests" in str(e).lower())
            )
            if is_rate_limit and attempt < max_retries:
                continue  # Retry
            # Not a rate limit error or exhausted retries
            err = f"{type(e).__name__}: {e}"
            # Try to capture llm0_messages even on error for debugging
            try:
                llm0_messages = tracker.get_messages(llm_id=0) if tracker is not None else None
            except Exception:
                llm0_messages = None
            break  # Exit retry loop

    seconds = time.time() - t0
    is_correct = exact_match(pred, gold) if err is None else False

    # Get tools used per round from tree_tracker
    tools_used_per_round: Optional[list[list[str]]] = None
    if tree_tracker is not None:
        try:
            tools_used_per_round = tree_tracker.get_tools_per_round()
        except Exception:
            tools_used_per_round = None

    # Get usage stats and rounds from tracker
    usage = tracker.usage if tracker is not None else None
    rounds = tracker.rounds if tracker is not None else None

    return EvalRecord(
        idx=idx,
        example_id=example_id,
        question=question,
        gold_answer=gold,
        pred_answer=pred,
        pred_raw=pred_raw,
        llm0_messages=llm0_messages,
        correct_em=is_correct,
        seconds=seconds,
        rounds=rounds,
        tools_used=tools_used_per_round,
        state_sequence=state_sequence,
        error=err,
        usage=usage,
    )


def worker_process(
    worker_id: int,
    examples: list[dict],
    config: EvalConfig,
    process_dir: Path,
    tool_funcs: list[Callable],
) -> None:
    """Worker process that evaluates a chunk of examples."""
    setup_env()

    # Configure BrowseComp search if needed (must be done in worker process)
    if "search" in config.tools or "get_document" in config.tools:
        configure_search(
            index_path="local/data/BrowseCompPlus/indexes/qwen3-embedding-8b/corpus.*.pkl",
            dataset_name="Tevatron/browsecomp-plus-corpus",
            sglang_url="http://localhost:30001",
            sglang_model="Qwen/Qwen3-Embedding-8B",
            task_prefix="Query: ",
        )

    # Create main model with custom chat_template_kwargs for thinking mode
    main_chat_template_kwargs = {"enable_thinking": config.enable_thinking} if config.model_provider == "local" else None
    main_model = create_model(
        provider=config.model_provider,
        model_type=config.model_type,
        model_url=config.model_url,
        stream=config.stream,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        chat_template_kwargs=main_chat_template_kwargs,
    )

    # Create search model (always disable thinking for search)
    non_thinking_model = create_model(
        provider=config.model_provider,
        model_type=config.model_type,
        model_url=config.model_url,
        stream=config.stream,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    # Create thinking model (always enable thinking for search)
    thinking_model = create_model(
        provider=config.model_provider,
        model_type=config.model_type,
        model_url=config.model_url,
        stream=config.stream,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        chat_template_kwargs={"enable_thinking": True} if config.model_provider == "local" else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=True)
    tools = [FunctionTool(func) for func in tool_funcs]
    # Add BrowseComp tools after configuration
    if "search" in config.tools:
        tools.append(FunctionTool(search))
    if "get_document" in config.tools:
        tools.append(FunctionTool(get_document))
    output_file = process_dir / f"worker_{worker_id}.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            rec = evaluate_single(
                idx=ex["_idx"],
                ex=ex,
                config=config,
                model=main_model,
                search_model=non_thinking_model,
                think_model=thinking_model,
                tokenizer=tokenizer,
                tools=tools,
            )
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f.flush()


def read_worker_records(process_dir: Path) -> list[dict]:
    """Read all records from worker files."""
    records: list[dict] = []
    for worker_file in sorted(process_dir.glob("worker_*.jsonl")):
        with worker_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


JUDGE_FIELDS = ["idx", "example_id", "question", "gold_answer", "pred_answer",
                "correct_em", "correct_llm", "judge_confidence", "judge_reason", "error_category"]


def read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_jsonl_path(config: EvalConfig) -> Path:
    """Get JSONL path from config output path."""
    return config.output.with_suffix(".jsonl") if config.output.suffix == ".json" else config.output


def run_llm_judge(jsonl_path: Path, config: EvalConfig, max_workers: int = 32) -> tuple[int, int, dict[str, list]]:
    """Run LLM judge for answer correctness and error categorization."""
    records = read_jsonl(jsonl_path)

    # Create judge model (local uses thinking by default)
    setup_env()
    judge_model = create_model(
        provider=config.judge_model_provider,
        model_type=config.judge_model_type or config.model_type,
        model_url=config.judge_api_url or config.model_url,
        # LLMJudge uses ChatAgent internally, which doesn't support streaming
        # model backends (Stream / generator responses). Keep judge non-streaming
        # even if the main evaluation run uses streaming.
        stream=False,
        max_tokens=config.max_tokens,
        chat_template_kwargs={"enable_thinking": True} if config.judge_model_provider == "local" else None,
    )
    judge = LLMJudge(judge_model, max_workers=max_workers)

    # Run judge with progress
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Judging answers", total=1)
        records, total_llm, correct_llm = judge.judge_batch(
            records, progress_callback=lambda c, t: progress.update(task, completed=c, total=t))

        task = progress.add_task("Categorizing errors", total=1)
        records, category_counts = judge.categorize_batch(
            records, progress_callback=lambda c, t: progress.update(task, completed=c, total=t))

    # Write results
    write_jsonl(jsonl_path, records)

    # Write judge-only results
    judge_path = jsonl_path.with_name(jsonl_path.stem + "_judge.jsonl")
    write_jsonl(judge_path, [{k: r.get(k) for k in JUDGE_FIELDS} for r in records])
    print(f"Judge results: {judge_path}")

    return total_llm, correct_llm, category_counts


def calculate_avg_usage(records: list[dict]) -> dict:
    """Calculate average usage stats from records.

    Returns:
        Dict with 'total', 'prompt', 'completion', 'cached', 'rounds' average values.
    """
    total_tokens_sum = 0
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    cached_tokens_sum = 0
    usage_count = 0
    rounds_sum = 0
    rounds_count = 0

    for rec in records:
        usage = rec.get("usage")
        if usage:
            total_tokens_sum += usage.get("total_tokens", 0)
            prompt_tokens_sum += usage.get("prompt_tokens", 0)
            completion_tokens_sum += usage.get("completion_tokens", 0)
            cached_tokens_sum += usage.get("cached_tokens", 0)
            usage_count += 1
        rounds = rec.get("rounds")
        if rounds is not None:
            rounds_sum += rounds
            rounds_count += 1

    if usage_count == 0:
        return {"total": 0, "prompt": 0, "completion": 0, "cached": 0, "rounds": 0}

    return {
        "total": total_tokens_sum / usage_count,
        "prompt": prompt_tokens_sum / usage_count,
        "completion": completion_tokens_sum / usage_count,
        "cached": cached_tokens_sum / usage_count,
        "rounds": rounds_sum / rounds_count if rounds_count > 0 else 0,
    }


def print_summary(
    total: int,
    correct_em: int,
    total_llm: int,
    correct_llm: int,
    category_counts: dict,
    avg_usage: Optional[dict] = None,
) -> None:
    """Print evaluation summary."""
    acc_em = (correct_em / total) if total else 0.0
    acc_llm = (correct_llm / total_llm) if total_llm else 0.0
    errors = total - total_llm
    print(f"\nDone. total={total}, finished={total_llm}, errors={errors}")
    print(f"  Exact Match: EM={correct_em} acc={acc_em:.3f}")
    print(f"  LLM Judge: correct={correct_llm} acc={acc_llm:.3f}")
    if avg_usage:
        print(f"\nAverage Usage per Question:")
        print(f"  Rounds: {avg_usage['rounds']:.1f}")
        print(f"  Total tokens: {avg_usage['total']:.1f}")
        print(f"  Prompt tokens: {avg_usage['prompt']:.1f} (cached: {avg_usage['cached']:.1f})")
        print(f"  Completion tokens: {avg_usage['completion']:.1f}")
    _print_error_table(category_counts)


def write_output_files(jsonl_path: Path, output_path: Path, output_format: str) -> None:
    """Write final output files in requested format and CSV."""
    # Convert to JSON array if needed
    if output_format == "json":
        _jsonl_to_json_array(jsonl_path, output_path)

    # Write CSV
    csv_path = output_path.with_suffix(".csv")
    csv_fields = ["idx", "example_id", "question", "gold_answer", "pred_answer", "pred_raw",
                  "correct_em", "correct_llm", "judge_confidence", "judge_reason",
                  "error_category", "rounds", "tools_used", "state_sequence", "seconds", "error", "usage"]

    with jsonl_path.open("r", encoding="utf-8") as fin, csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=csv_fields)
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                row = {k: obj.get(k) for k in csv_fields}
                # Convert list/dict fields to JSON strings for CSV
                if row.get("tools_used") is not None:
                    row["tools_used"] = json.dumps(row["tools_used"])
                if row.get("state_sequence") is not None:
                    row["state_sequence"] = json.dumps(row["state_sequence"])
                if row.get("usage") is not None:
                    row["usage"] = json.dumps(row["usage"])
                writer.writerow(row)
            except json.JSONDecodeError:
                continue


def _jsonl_to_json_array(jsonl_path: Path, json_path: Path) -> None:
    """Convert JSONL into a pretty-printed JSON array."""
    first = True
    with jsonl_path.open("r", encoding="utf-8") as fin, json_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not first:
                fout.write(",\n")
            first = False
            obj_str = json.dumps(obj, ensure_ascii=False, indent=2)
            fout.write("\n".join("  " + s for s in obj_str.splitlines()))
        fout.write("\n]\n")


def load_dataset_examples(config: EvalConfig) -> list[dict]:
    """Load examples from HotpotQA or BrowseComp dataset."""
    if config.dataset == "hotpotqa":
        split = config.split
        if config.limit is not None and config.limit > 0:
            split = f"{split}[:{config.limit}]"
        ds = load_dataset("hotpotqa/hotpot_qa", config.subset, split=split)
        examples = []
        for idx, ex in enumerate(ds):
            ex_dict = dict(ex)
            ex_dict["_idx"] = idx
            examples.append(ex_dict)
        return examples
    elif config.dataset == "browsecomp":
        if config.data_path is None:
            config.data_path = "local/data/BrowseCompPlus/data/browsecomp_qa_pairs.jsonl"
        data_path = Path(config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"BrowseComp data file not found: {data_path}")
        examples = []
        with data_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if config.limit is not None and config.limit > 0 and idx >= config.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    # Map browsecomp fields to common format
                    ex["question"] = ex.pop("query")
                    ex["_idx"] = idx
                    examples.append(ex)
                except json.JSONDecodeError:
                    continue
        return examples
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


def _print_error_table(category_counts: dict[str, list]) -> None:
    """Print error category statistics with rich table."""
    console = Console()
    console.print("\n[bold]ERROR CATEGORY ANALYSIS[/bold]\n")

    total_incorrect = sum(len(examples) for examples in category_counts.values())

    table = Table(show_header=True, header_style="bold cyan", title="Error Distribution")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Category", style="magenta", min_width=30)
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    sorted_categories = sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)

    for rank, (category, examples) in enumerate(sorted_categories, 1):
        count = len(examples)
        percentage = (count / total_incorrect * 100) if total_incorrect > 0 else 0
        table.add_row(f"{rank}", category, f"{count}/{total_incorrect}", f"{percentage:.1f}%")

    console.print(table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "browsecomp"],
                        help="Dataset to evaluate: hotpotqa or browsecomp")
    parser.add_argument("--data-path", type=str, default="local/data/BrowseCompPlus/data/browsecomp_qa_pairs.jsonl",
                        help="Path to data file (for browsecomp, default: local/data/BrowseCompPlus/data/browsecomp_qa_pairs.jsonl)")
    parser.add_argument("--subset", default="distractor", choices=["distractor", "fullwiki"], help="HotpotQA subset (only for hotpotqa dataset)")
    parser.add_argument("--split", default="validation", help="HotpotQA split (only for hotpotqa dataset)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--output", default="local/evaluation/direct/hotpotqa.jsonl")
    parser.add_argument("--output-format", default="json", choices=["jsonl", "json"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-type", default=None,
                        help="Model type (default: local='local', openai='gpt-4o-mini', gemini='gemini-3-flash-preview')")
    parser.add_argument("--model-provider", default="local", choices=["local", "openai", "gemini", "fireworks"],
                        help="Model provider: local (OpenAI-compatible), openai, or gemini")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    parser.add_argument("--mode", default="oneflow", choices=["oneflow", "single", "tree", "tool", "singletool"])
    parser.add_argument("--main_to_search", type=str, default="none", choices=["all", "initial", "none"])
    parser.add_argument("--search", type=str, default="none", choices=["all", "initial", "none"])
    parser.add_argument("--search_to_main", type=str, default="qr", choices=["all", "qr"])
    parser.add_argument("--main", type=str, default="all", choices=["all", "none", "qr"])
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["search_engine"],
        choices=["search_engine", "search_wiki", "search_google", "search_tavily", "search", "get_document"],
        help="Tools to enable (default: search_engine). Use 'search' and 'get_document' for BrowseComp.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--state-rule",
        nargs="+",
        default=["execute", "plan", "rewind", "answer", "think"],
        help="Allowed tree actions (default: execute plan rewind answer)",
    )
    parser.add_argument("--judge-only", action="store_true",
                        help="Only run LLM judge on existing data (use with --output to specify input file)")
    parser.add_argument("--judge-model-provider", type=str, default="local", choices=["local", "openai", "gemini", "fireworks"],
                        help="Model provider for LLM judge (default: local)")
    parser.add_argument("--judge-api-url", type=str, default=None, help="API URL for LLM judge (defaults to model-url)")
    parser.add_argument("--judge-model-type", type=str, default=None, help="Model type for LLM judge")
    parser.add_argument("--step-timeout", type=float, default=None, help="Timeout in seconds for agent step calls")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for main agent (local models only)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming (adds {'stream': True} to model_config_dict)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum tokens to generate per model call (default: 32768)")
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Re-run only examples with non-empty 'error' in existing output JSONL",
    )
    parser.add_argument("--history-assistant", type=str, default="full", choices=["full", "none", "summarized"],
                        help="History config for assistant messages in singletool mode (default: full)")
    parser.add_argument("--history-tool", type=str, default="full", choices=["full", "none", "summarized"],
                        help="History config for tool messages in singletool mode (default: full)")
    parser.add_argument("--history-reasoning", type=str, default="full", choices=["full", "none", "summarized"],
                        help="History config for reasoning messages in singletool mode (default: full)")
    parser.add_argument("--history-delay", type=int, default=0,
                        help="Delay in rounds before applying history transformations (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for model generation (default: 0.0)")
    args = parser.parse_args()

    config = EvalConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        max_rounds=args.max_rounds,
        output=Path(args.output),
        output_format=args.output_format,
        resume=args.resume,
        model_url=args.model_url,
        model_type=args.model_type,
        model_provider=args.model_provider,
        tokenizer=args.tokenizer,
        mode=args.mode,
        main_to_search=args.main_to_search,
        search=args.search,
        search_to_main=args.search_to_main,
        main=args.main,
        tools=args.tools,
        num_workers=args.num_workers,
        state_rule=args.state_rule,
        judge_model_provider=args.judge_model_provider,
        judge_api_url=args.judge_api_url,
        judge_model_type=args.judge_model_type,
        step_timeout=args.step_timeout,
        enable_thinking=args.enable_thinking,
        stream=args.stream,
        patch=args.patch,
        max_tokens=args.max_tokens,
        history_assistant=args.history_assistant,
        history_tool=args.history_tool,
        history_reasoning=args.history_reasoning,
        history_delay=args.history_delay,
        temperature=args.temperature,
    )

    config.output.parent.mkdir(parents=True, exist_ok=True)

    # Judge-only mode: skip research, just run LLM judge on existing data
    if args.judge_only:
        jsonl_path = get_jsonl_path(config)
        if not jsonl_path.exists():
            print(f"Error: Input file not found: {jsonl_path}")
            return

        print(f"Running judge-only mode on: {jsonl_path}")
        total_llm, correct_llm, category_counts = run_llm_judge(
            jsonl_path,
            config,
            max_workers=config.num_workers,
        )

        records = read_jsonl(jsonl_path)
        total = len(records)
        correct_em = sum(1 for r in records if r.get("correct_em", False))
        avg_usage = calculate_avg_usage(records)

        write_output_files(jsonl_path, config.output, config.output_format)
        print_summary(total, correct_em, total_llm, correct_llm, category_counts, avg_usage)
        return

    process_dir = config.output.parent / "process"
    process_dir.mkdir(exist_ok=True)

    # Load dataset
    all_examples = load_dataset_examples(config)

    # Patch mode: re-run only examples that failed previously
    existing_records: list[dict] | None = None
    if config.patch:
        jsonl_path = get_jsonl_path(config)
        if not jsonl_path.exists():
            print(f"Error: Input file not found for patch mode: {jsonl_path}")
            return
        existing_records = read_jsonl(jsonl_path)
        error_ids = {
            str(r.get("example_id"))
            for r in existing_records
            if r.get("error")
        }
        if not error_ids:
            print("No failed examples found to patch.")
            return
        examples = [ex for ex in all_examples if str(ex["id"]) in error_ids]
    else:
        # Skip already done (optional)
        done_ids: set[str] = set()
        if config.resume:
            # Check existing worker files
            for worker_file in process_dir.glob("worker_*.jsonl"):
                worker_done = load_done_ids(worker_file)
                done_ids.update(worker_done)

        # Filter dataset
        examples = [ex for ex in all_examples if str(ex["id"]) not in done_ids]

    if not examples:
        print("No new examples to evaluate.")
        return

    # Define tools as list of callables
    # Note: search and get_document are configured and added in worker_process
    tool_funcs: list[Callable] = []
    search_toolkit: Optional[SearchToolkit] = None
    for tool_name in config.tools:
        if tool_name == "search_engine":
            tool_funcs.append(search_engine)
        elif tool_name in ("search", "get_document"):
            # These are handled in worker_process after configure_search
            pass
        elif tool_name == "search_wiki":
            if search_toolkit is None:
                search_toolkit = SearchToolkit()
            tool_funcs.append(search_toolkit.search_wiki)
        elif tool_name == "search_google":
            if search_toolkit is None:
                search_toolkit = SearchToolkit()
            tool_funcs.append(search_toolkit.search_google)
        elif tool_name == "search_tavily":
            if search_toolkit is None:
                search_toolkit = SearchToolkit()
            tool_funcs.append(search_toolkit.search_tavily)
        else:
            raise ValueError(f"Unsupported tool: {tool_name}")

    # Split examples into chunks
    num_workers = min(config.num_workers, len(examples))
    chunk_size = (len(examples) + num_workers - 1) // num_workers
    chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]

    # Start workers
    processes = []
    for worker_id, chunk in enumerate(chunks):
        p = mp.Process(target=worker_process, args=(worker_id, chunk, config, process_dir, tool_funcs))
        p.start()
        processes.append(p)

    # Monitor with Rich Progress (file-based)
    worker_files = [process_dir / f"worker_{i}.jsonl" for i in range(len(chunks))]

    # Group workers if more than 10 for cleaner display
    max_display_groups = 10
    num_actual_workers = len(chunks)
    if num_actual_workers > max_display_groups:
        # Create groups of workers
        group_size = (num_actual_workers + max_display_groups - 1) // max_display_groups
        worker_groups = []
        for g in range(max_display_groups):
            start_idx = g * group_size
            end_idx = min(start_idx + group_size, num_actual_workers)
            if start_idx < num_actual_workers:
                worker_groups.append(list(range(start_idx, end_idx)))
    else:
        # Each worker is its own group
        worker_groups = [[i] for i in range(num_actual_workers)]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        # Create progress bars for groups
        group_tasks = []
        group_totals = []
        for g_idx, group in enumerate(worker_groups):
            group_total = sum(len(chunks[w]) for w in group)
            group_totals.append(group_total)
            if len(worker_groups) == num_actual_workers:
                label = f"Worker {g_idx}"
            else:
                label = f"Group {g_idx} ({len(group)} workers)"
            group_tasks.append(progress.add_task(label, total=group_total))
        overall = progress.add_task("[bold]Overall", total=len(examples))

        # Poll worker files
        while any(p.is_alive() for p in processes):
            # Count completions per worker
            worker_counts = []
            for wfile in worker_files:
                if wfile.exists():
                    worker_counts.append(sum(1 for _ in wfile.open("r") if _.strip()))
                else:
                    worker_counts.append(0)
            # Update group progress
            for g_idx, group in enumerate(worker_groups):
                group_completed = sum(worker_counts[w] for w in group)
                progress.update(group_tasks[g_idx], completed=group_completed)
            progress.update(overall, completed=sum(worker_counts))
            time.sleep(0.5)

        # Final update
        worker_counts = []
        for wfile in worker_files:
            if wfile.exists():
                worker_counts.append(sum(1 for _ in wfile.open("r") if _.strip()))
            else:
                worker_counts.append(0)
        for g_idx, group in enumerate(worker_groups):
            group_completed = sum(worker_counts[w] for w in group)
            progress.update(group_tasks[g_idx], completed=group_completed)
        progress.update(overall, completed=sum(worker_counts))

    for p in processes:
        p.join()

    # Aggregate results
    jsonl_path = get_jsonl_path(config)
    worker_records = read_worker_records(process_dir)
    if config.patch:
        existing_by_id = {str(r.get("example_id")): r for r in (existing_records or [])}
        for rec in worker_records:
            existing_by_id[str(rec.get("example_id"))] = rec
        merged_records = list(existing_by_id.values())
    else:
        merged_records = worker_records

    write_jsonl(jsonl_path, merged_records)
    total = len(merged_records)
    correct_em = sum(1 for r in merged_records if r.get("correct_em", False))

    # LLM judge for answer correctness and error categorization
    print("\nRunning LLM judge...")
    total_llm, correct_llm, category_counts = run_llm_judge(
        jsonl_path,
        config,
        max_workers=config.num_workers,
    )

    # Write final output files
    write_output_files(jsonl_path, config.output, config.output_format)

    # Write summary with configuration and prompts
    acc_em = (correct_em / total) if total else 0.0
    acc_llm = (correct_llm / total_llm) if total_llm else 0.0
    csv_path = config.output.with_suffix(".csv")

    # Calculate average usage stats
    avg_usage = calculate_avg_usage(merged_records)

    errors = total - total_llm
    summary_lines = [
        "=" * 80,
        "EVALUATION SUMMARY",
        "=" * 80,
        f"Results: total={total}, finished={total_llm}, errors={errors}",
        f"  Exact Match: EM={correct_em} acc={acc_em:.3f}",
        f"  LLM Eval: correct={correct_llm} acc={acc_llm:.3f}",
        f"Output: {config.output}",
        f"CSV: {csv_path}",
        "",
        "Average Usage per Question:",
        f"  Rounds: {avg_usage['rounds']:.1f}",
        f"  Total tokens: {avg_usage['total']:.1f}",
        f"  Prompt tokens: {avg_usage['prompt']:.1f} (cached: {avg_usage['cached']:.1f})",
        f"  Completion tokens: {avg_usage['completion']:.1f}",
        "",
        "Arguments:",
        f"  --dataset {config.dataset}",
        f"  --data-path {config.data_path}",
        f"  --subset {config.subset}",
        f"  --split {config.split}",
        f"  --limit {config.limit}",
        f"  --max-rounds {config.max_rounds}",
        f"  --output {config.output}",
        f"  --output-format {config.output_format}",
        f"  --resume {config.resume}",
        f"  --model-url {config.model_url}",
        f"  --model-type {config.model_type}",
        f"  --model-provider {config.model_provider}",
        f"  --tokenizer {config.tokenizer}",
        f"  --mode {config.mode}",
        f"  --main_to_search {config.main_to_search}",
        f"  --search {config.search}",
        f"  --search_to_main {config.search_to_main}",
        f"  --main {config.main}",
        f"  --num-workers {config.num_workers}",
        f"  --state-rule {' '.join(config.state_rule)}",
        f"  --temperature {config.temperature}",
        "",
    ]

    # Prompt definitions per mode
    mode_prompts = {}
    if config.mode == "tree":
        from rosetta.workflow.tree_prompt import INIT_PROMPT, WORKER_PROMPT, REWIND_PROMPT
        mode_prompts["tree"] = [
            ("INIT_PROMPT", INIT_PROMPT),
            ("WORKER_PROMPT", WORKER_PROMPT),
            ("REWIND_PROMPT", REWIND_PROMPT),
        ]
    elif config.mode == "oneflow":
        from rosetta.workflow.prompt import (
            SEARCH_TASK_DECOMPOSE_PROMPT,
            TASK_REVISE_PROMPT,
            FORCE_ANSWER_PROMPT,
            SEARCH_AGENT_PROMPT,
        )
        mode_prompts["oneflow"] = [
            ("SEARCH_TASK_DECOMPOSE_PROMPT", SEARCH_TASK_DECOMPOSE_PROMPT),
            ("TASK_REVISE_PROMPT", TASK_REVISE_PROMPT),
            ("FORCE_ANSWER_PROMPT", FORCE_ANSWER_PROMPT),
            ("SEARCH_AGENT_PROMPT", SEARCH_AGENT_PROMPT),
        ]

    # Add prompts to summary
    if config.mode in mode_prompts:
        summary_lines.append(f"Prompts ({config.mode.capitalize()} Mode):")
        summary_lines.append("")
        for prompt_name, prompt_text in mode_prompts[config.mode]:
            summary_lines.extend([
                f"{prompt_name}:",
                "-" * 80,
                prompt_text,
                "",
            ])

    # Add error category statistics to summary
    summary_lines.append("Error Category Analysis:")
    summary_lines.append("")
    summary_lines.append(f"{'Rank':<6} {'Category':<35} {'Count':<15} {'Percentage':<10}")
    summary_lines.append("-" * 80)

    total_incorrect = sum(len(examples) for examples in category_counts.values())
    sorted_categories = sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)

    for rank, (category, examples) in enumerate(sorted_categories, 1):
        count = len(examples)
        percentage = (count / total_incorrect * 100) if total_incorrect > 0 else 0
        summary_lines.append(f"{rank:<6} {category:<35} {count}/{total_incorrect:<13} {percentage:.1f}%")

    summary_lines.append("")
    summary_lines.append("=" * 80)

    summary_path = config.output.parent / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Output: {config.output}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print_summary(total, correct_em, total_llm, correct_llm, category_counts, avg_usage)


if __name__ == "__main__":
    main()
