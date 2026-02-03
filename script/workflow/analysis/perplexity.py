#!/usr/bin/env python3
"""
Perplexity and entropy analysis for conversation histories.

Loads evaluation results, prefills sequences through a model, and analyzes
how entropy/perplexity varies by token position and message type.

Usage:
    # Local HuggingFace model
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b

    # Fireworks API
    python script/workflow/analysis/perplexity.py \
        --input local/evaluation/results.jsonl \
        --model openai/gpt-oss-20b \
        --backend fireworks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from camel.toolkits import FunctionTool

from rosetta.workflow.analysis.interface import (
    AggregatedMetrics,
    AnalysisResult,
    aggregate_results,
    analyze_conversation,
    extract_conversations,
    load_evaluation_results,
    save_token_plot_data_csv,
)
from rosetta.utils.core import (
    EstimatedEntropyUnified,
    NegLogProbUnified,
    PerplexityUnified,
    TopKEntropyUnified,
    TopKMassUnified,
)
from rosetta.workflow.analysis.oss_tokenizer import batch_tokenize_with_sections
from rosetta.workflow.analysis.plot import (
    plot_metric_by_position,
    plot_metric_by_role,
    plot_metric_distribution,
)
from rosetta.workflow.browse_searcher import get_document, search


# =============================================================================
# Output
# =============================================================================


def save_results(results: List[AnalysisResult], output_path: Path):
    """Save analysis results to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            data = {
                "conversation_id": result.conversation_id,
                "token_count": result.token_count,
                "overall_metrics": result.overall_metrics,
                "sections": [
                    {
                        "role": s.role,
                        "content_type": s.content_type,
                        "token_count": s.token_count,
                        "metrics": s.metrics,
                    }
                    for s in result.sections
                ],
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


def save_aggregated(aggregated: AggregatedMetrics, output_path: Path):
    """Save aggregated metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "num_conversations": aggregated.num_conversations,
        "total_tokens": aggregated.total_tokens,
        "by_role": aggregated.by_role,
        "by_content_type": aggregated.by_content_type,
        "overall": aggregated.overall,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregated metrics to {output_path}")


def print_summary(aggregated: AggregatedMetrics):
    """Print a summary of aggregated metrics."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Conversations: {aggregated.num_conversations}")
    print(f"Total tokens: {aggregated.total_tokens}")

    print("\nOverall Metrics:")
    for metric, value in aggregated.overall.items():
        print(f"  {metric}: {value:.4f}")

    print("\nMetrics by Role:")
    for role, metrics in sorted(aggregated.by_role.items()):
        print(f"  {role}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze perplexity and entropy of conversation histories."
    )
    parser.add_argument(
        "--input", "-i",
        default="local/evaluation/gpt_oss_20b/singletool/browsecomp/full_full_full/results.jsonl",
        help="Path to evaluation results JSONL file",
    )
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-oss-20b",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="local/analysis/perplexity",
        help="Directory for output files",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "fireworks"],
        default="local",
        help="Backend: local (HuggingFace) or fireworks (API)",
    )
    parser.add_argument(
        "--fireworks-api-key",
        default=None,
        help="Fireworks API key (or set FIREWORKS_API_KEY env var)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of conversations to analyze",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (skip longer)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip generating CSV output",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include final message (default: exclude to include all reasoning)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load model for local backend
    model = None
    fireworks_model = None
    if args.backend == "local":
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
    else:
        # Convert HuggingFace model name to Fireworks format
        if args.model.startswith("accounts/"):
            fireworks_model = args.model
        else:
            model_name = args.model.split("/")[-1]
            fireworks_model = f"accounts/fireworks/models/{model_name}"
        print(f"Using Fireworks API with model: {fireworks_model}")

    # Load and tokenize conversations
    print(f"\nLoading: {input_path}")
    records = load_evaluation_results(input_path)
    if args.limit:
        records = records[:args.limit]
    conversations = extract_conversations(records)

    # Tool schemas for section detection
    default_tools = [
        FunctionTool(search).get_openai_tool_schema(),
        FunctionTool(get_document).get_openai_tool_schema(),
    ]

    tokenized = batch_tokenize_with_sections(
        conversations,
        tokenizer,
        tools=default_tools,
        max_length=args.max_length,
        show_progress=True,
        exclude_final=not args.include_final,
        convert_reasoning=True,
    )

    # Analyze each conversation
    print(f"\nAnalyzing {len(tokenized)} conversations...")
    results = []
    try:
        from rich.progress import track
        iterator = track(tokenized, description=f"Analyzing ({args.backend})...")
    except ImportError:
        iterator = tokenized

    for conv in iterator:
        try:
            result = analyze_conversation(
                conv,
                backend=args.backend,
                model=model,
                fireworks_model=fireworks_model,
                fireworks_api_key=args.fireworks_api_key,
            )
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to analyze {conv.conversation_id}: {e}")
            continue

    if not results:
        print("No results to analyze")
        return

    # Aggregate and output
    aggregated = aggregate_results(results)
    save_results(results, output_dir / "results.jsonl")
    save_aggregated(aggregated, output_dir / "aggregated.json")
    print_summary(aggregated)

    # CSV export
    if not args.no_csv:
        csv_metrics = [
            NegLogProbUnified(),
            PerplexityUnified(),
            EstimatedEntropyUnified(),  # Exact for HF, tight lower bound for API
            TopKEntropyUnified(),       # Renormalized top-k entropy (comparable across backends)
            TopKMassUnified(),
        ]
        save_token_plot_data_csv(
            results=results,
            metrics=csv_metrics,
            output_path=output_dir / "plot_data.csv",
        )

    # Plots
    if not args.no_plot:
        metric_names = list(results[0].overall_metrics.keys()) if results else []
        for metric in metric_names:
            plot_metric_by_role(aggregated, metric, output_dir / f"{metric}_by_role.png")
            plot_metric_distribution(results, metric, output_dir / f"{metric}_distribution.png")
            plot_metric_by_position(results, metric, output_dir / f"{metric}_by_position.png")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
