"""Interface for loading evaluation results and tokenizing with section tracking.

This module provides:
- Data loading from evaluation JSONL files
- Tokenization with section tracking (role, content_type)
- Unified backend abstraction for computing metrics (local HF or Fireworks API)
- Analysis result structures and aggregation
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Tuple

import torch

from rosetta.utils.core import (
    DEFAULT_UNIFIED_METRICS,
    EntropyLowerBoundUnified,
    EntropyUnified,
    EstimatedEntropyUnified,
    NegLogProbUnified,
    PerplexityUnified,
    PrefillResult,
    TokenMetric,
    TopKEntropyUnified,
    TopKMassUnified,
    UnifiedMetric,
    compute_unified_metrics,
    fireworks_prefill,
    hf_logits_to_prefill_result,
)


@dataclass
class TokenSection:
    """Tracks a contiguous section of tokens from one message.

    Attributes:
        start_idx: Start position in token sequence (inclusive).
        end_idx: End position in token sequence (exclusive).
        role: Message role ("system", "user", "assistant", "tool").
        message_idx: Index of source message in the conversation.
        content_type: Type of content ("text", "tool_call", "reasoning").
    """

    start_idx: int
    end_idx: int
    role: str
    message_idx: int
    content_type: str = "text"

    @property
    def length(self) -> int:
        """Number of tokens in this section."""
        return self.end_idx - self.start_idx

    def __repr__(self) -> str:
        return (
            f"TokenSection({self.role}:{self.content_type}, "
            f"[{self.start_idx}:{self.end_idx}], msg={self.message_idx})"
        )


@dataclass
class TokenizedConversation:
    """A conversation converted to tokens with section tracking.

    Attributes:
        input_ids: Token IDs as tensor [seq_len].
        sections: List of TokenSection metadata.
        messages: Original messages (for reference).
        conversation_id: Optional identifier for the conversation.
    """

    input_ids: torch.Tensor
    sections: List[TokenSection]
    messages: List[Dict[str, Any]]
    conversation_id: Optional[str] = None

    @property
    def seq_len(self) -> int:
        """Total sequence length."""
        return len(self.input_ids)

    def get_section_mask(self, role: Optional[str] = None, content_type: Optional[str] = None) -> torch.Tensor:
        """Create a boolean mask for tokens matching the filter criteria.

        Args:
            role: Filter by role (e.g., "assistant", "tool").
            content_type: Filter by content type (e.g., "text", "reasoning").

        Returns:
            Boolean tensor of shape [seq_len].
        """
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        for section in self.sections:
            if role is not None and section.role != role:
                continue
            if content_type is not None and section.content_type != content_type:
                continue
            mask[section.start_idx : section.end_idx] = True
        return mask

    def get_role_labels(self) -> torch.Tensor:
        """Create a tensor with role labels for each token.

        Returns:
            Integer tensor [seq_len] with role indices:
            0=system, 1=user, 2=assistant, 3=tool, -1=unknown
        """
        role_map = {"system": 0, "user": 1, "assistant": 2, "tool": 3}
        labels = torch.full((self.seq_len,), -1, dtype=torch.long)
        for section in self.sections:
            labels[section.start_idx : section.end_idx] = role_map.get(section.role, -1)
        return labels


@dataclass
class TransformResult:
    """Result of applying a context transformation.

    Attributes:
        original: The original tokenized conversation.
        transformed: The transformed tokenized conversation.
        transform_name: Name of the transformation applied.
        token_mapping: Optional mapping from transformed token indices to original.
    """

    original: TokenizedConversation
    transformed: TokenizedConversation
    transform_name: str
    token_mapping: Optional[Dict[int, int]] = None


# =============================================================================
# Data Loading
# =============================================================================


def load_evaluation_results(path: Path | str) -> List[Dict[str, Any]]:
    """Load evaluation records from a JSONL file.

    Args:
        path: Path to the JSONL file (e.g., results.jsonl).

    Returns:
        List of record dictionaries.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue
    return records


def extract_conversations(
    records: List[Dict[str, Any]],
    message_field: str = "llm0_messages",
    id_field: str = "example_id",
    filter_errors: bool = True,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Extract conversations from evaluation records.

    Args:
        records: List of evaluation record dicts.
        message_field: Field name containing the message list.
        id_field: Field name for the conversation ID.
        filter_errors: If True, skip records with non-null error field.

    Returns:
        List of (conversation_id, messages) tuples.
    """
    conversations = []
    for rec in records:
        if filter_errors and rec.get("error"):
            continue
        messages = rec.get(message_field)
        if not messages or not isinstance(messages, list):
            continue
        conv_id = str(rec.get(id_field, rec.get("idx", len(conversations))))
        conversations.append((conv_id, messages))
    return conversations


# =============================================================================
# Tokenization with Section Tracking
# =============================================================================


def _get_message_parts(msg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract content parts from a message.

    Returns a list of (content_type, text) tuples for the message.
    """
    parts = []
    role = msg.get("role", "unknown")

    # Handle reasoning content (assistant only)
    if role == "assistant":
        reasoning = msg.get("reasoning_content") or msg.get("_reasoning")
        if reasoning:
            parts.append(("reasoning", str(reasoning)))

    # Main content
    content = msg.get("content", "")
    if content:
        parts.append(("text", str(content)))

    # Tool calls (assistant only)
    if role == "assistant":
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc_text = json.dumps(tool_calls, ensure_ascii=False)
            parts.append(("tool_call", tc_text))

    # If no parts extracted, add empty text
    if not parts:
        parts.append(("text", ""))

    return parts


def tokenize_conversation(
    messages: List[Dict[str, Any]],
    tokenizer,
    add_generation_prompt: bool = False,
    conversation_id: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> TokenizedConversation:
    """Convert a conversation to tokens with section tracking.

    This tokenizes each message individually to track section boundaries,
    then concatenates them to form the full sequence.

    Args:
        messages: List of message dicts with "role" and "content".
        tokenizer: HuggingFace tokenizer with chat template support.
        add_generation_prompt: If True, add generation prompt at the end.
        conversation_id: Optional identifier for the conversation.
        tools: Optional list of tool schemas (OpenAI format). Can be:
            - List of dicts with OpenAI tool schema format
            - List of FunctionTool objects (will call get_openai_tool_schema())

    Returns:
        TokenizedConversation with input_ids and section metadata.
    """
    # Convert FunctionTool objects to schemas if needed
    tool_schemas = None
    if tools:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif hasattr(tool, 'get_openai_tool_schema'):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
    sections = []
    all_input_ids = []
    current_pos = 0

    # Try to use chat template for full conversation first to get proper formatting
    # But we need per-message boundaries, so we tokenize incrementally
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")

        # Build a minimal conversation up to this message to get proper formatting
        # This handles chat templates that add special tokens between messages
        prefix_msgs = messages[:msg_idx]
        current_msgs = messages[: msg_idx + 1]

        # Tokenize prefix (all messages before current)
        if prefix_msgs:
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=False,
                tools=tool_schemas,
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        else:
            prefix_ids = []

        # Tokenize up to and including current message
        current_text = tokenizer.apply_chat_template(
            current_msgs, tokenize=False, add_generation_prompt=False,
            tools=tool_schemas,
        )
        current_ids = tokenizer.encode(current_text, add_special_tokens=False)

        # The tokens for this message are the difference
        msg_start = len(prefix_ids)
        msg_end = len(current_ids)
        msg_ids = current_ids[msg_start:]

        if msg_ids:
            # Create section for this message
            # For now, treat entire message as single section
            # Future: split by content parts (reasoning, text, tool_call)
            section = TokenSection(
                start_idx=current_pos,
                end_idx=current_pos + len(msg_ids),
                role=role,
                message_idx=msg_idx,
                content_type="text",  # TODO: split by parts
            )
            sections.append(section)
            all_input_ids.extend(msg_ids)
            current_pos += len(msg_ids)

    # Add generation prompt if requested
    if add_generation_prompt and messages:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tool_schemas,
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) > current_pos:
            # There's a generation prompt suffix
            gen_ids = full_ids[current_pos:]
            section = TokenSection(
                start_idx=current_pos,
                end_idx=current_pos + len(gen_ids),
                role="assistant",
                message_idx=len(messages),  # Virtual message index
                content_type="generation_prompt",
            )
            sections.append(section)
            all_input_ids.extend(gen_ids)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=messages,
        conversation_id=conversation_id,
    )


def tokenize_conversation_simple(
    messages: List[Dict[str, Any]],
    tokenizer,
    conversation_id: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> TokenizedConversation:
    """Simplified tokenization that tokenizes the full conversation at once.

    This is faster but may have less accurate section boundaries for some
    tokenizers with complex chat templates.

    Args:
        messages: List of message dicts.
        tokenizer: HuggingFace tokenizer.
        conversation_id: Optional identifier.
        tools: Optional list of tool schemas (OpenAI format) or FunctionTool objects.

    Returns:
        TokenizedConversation with approximate section boundaries.
    """
    # Convert FunctionTool objects to schemas if needed
    tool_schemas = None
    if tools:
        tool_schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_schemas.append(tool)
            elif hasattr(tool, 'get_openai_tool_schema'):
                tool_schemas.append(tool.get_openai_tool_schema())
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

    # Tokenize full conversation
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=tool_schemas
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").squeeze(0)

    # Estimate section boundaries by tokenizing each message's content
    sections = []
    current_pos = 0

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Tokenize just the content to estimate its length
        if content:
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            content_len = len(content_ids)
        else:
            content_len = 0

        # Account for role tokens and template formatting (approximate)
        # This is a rough estimate; exact boundaries depend on the chat template
        estimated_overhead = 10  # Typical overhead for role markers, etc.

        section_len = content_len + estimated_overhead
        section_end = min(current_pos + section_len, len(input_ids))

        if section_end > current_pos:
            section = TokenSection(
                start_idx=current_pos,
                end_idx=section_end,
                role=role,
                message_idx=msg_idx,
                content_type="text",
            )
            sections.append(section)
            current_pos = section_end

    # Adjust last section to cover remaining tokens
    if sections and current_pos < len(input_ids):
        sections[-1] = TokenSection(
            start_idx=sections[-1].start_idx,
            end_idx=len(input_ids),
            role=sections[-1].role,
            message_idx=sections[-1].message_idx,
            content_type=sections[-1].content_type,
        )

    return TokenizedConversation(
        input_ids=input_ids,
        sections=sections,
        messages=messages,
        conversation_id=conversation_id,
    )


# =============================================================================
# Context Transformations
# =============================================================================


def apply_context_transform(
    conversation: TokenizedConversation,
    transform_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    tokenizer,
    transform_name: str = "transform",
) -> TransformResult:
    """Apply a transformation to the conversation and re-tokenize.

    Args:
        conversation: Original tokenized conversation.
        transform_fn: Function that transforms the message list.
        tokenizer: Tokenizer for re-tokenizing after transformation.
        transform_name: Name of the transformation for logging.

    Returns:
        TransformResult with original and transformed conversations.
    """
    # Apply transformation to messages
    transformed_messages = transform_fn(conversation.messages)

    # Re-tokenize
    transformed_conv = tokenize_conversation(
        transformed_messages,
        tokenizer,
        conversation_id=f"{conversation.conversation_id}__{transform_name}",
    )

    return TransformResult(
        original=conversation,
        transformed=transformed_conv,
        transform_name=transform_name,
        token_mapping=None,  # Token alignment is complex after transforms
    )


def create_summarize_transform(
    model,
    summarize_tool: bool = True,
    summarize_assistant: bool = False,
    summarize_reasoning: bool = False,
):
    """Create a transformation function that summarizes content.

    Args:
        model: CAMEL model backend for summarization.
        summarize_tool: Whether to summarize tool responses.
        summarize_assistant: Whether to summarize assistant responses.
        summarize_reasoning: Whether to summarize reasoning content.

    Returns:
        A callable that transforms a message list.
    """
    from rosetta.workflow.contextManage import (
        inject_call_context,
        summarize_content,
        summarize_reasoning as summarize_reasoning_fn,
        summarize_tool_resp,
    )

    def transform(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Make a copy to avoid modifying original
        messages = [dict(m) for m in messages]

        # Inject call context for tool messages
        inject_call_context(messages)

        result = []
        for msg in messages:
            role = msg.get("role", "")

            if role == "tool" and summarize_tool:
                summarized = summarize_tool_resp([msg], model)
                result.append(summarized[0])
            elif role == "assistant":
                new_msg = dict(msg)
                if summarize_assistant and msg.get("content"):
                    summarized = summarize_content([msg], model)
                    new_msg["content"] = summarized[0].get("content", "")
                if summarize_reasoning and (msg.get("_reasoning") or msg.get("reasoning_content")):
                    reasoning_msg = {"content": msg.get("_reasoning") or msg.get("reasoning_content")}
                    summarized = summarize_reasoning_fn([reasoning_msg], model)
                    if "_reasoning" in msg:
                        new_msg["_reasoning"] = summarized[0].get("content", "")
                    if "reasoning_content" in msg:
                        new_msg["reasoning_content"] = summarized[0].get("content", "")
                result.append(new_msg)
            else:
                result.append(msg)

        return result

    return transform


# =============================================================================
# Batch Processing Utilities
# =============================================================================


def batch_tokenize_conversations(
    conversations: List[Tuple[str, List[Dict[str, Any]]]],
    tokenizer,
    max_length: Optional[int] = None,
    show_progress: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[TokenizedConversation]:
    """Tokenize multiple conversations.

    Args:
        conversations: List of (id, messages) tuples.
        tokenizer: HuggingFace tokenizer.
        max_length: Optional maximum sequence length (skip longer).
        show_progress: Whether to show progress bar.
        tools: Optional list of tool schemas (OpenAI format) or FunctionTool objects.
            Applied to all conversations in the batch.

    Returns:
        List of TokenizedConversation objects.
    """
    results = []

    iterator = conversations
    if show_progress:
        try:
            from rich.progress import track
            iterator = track(conversations, description="Tokenizing...")
        except ImportError:
            pass

    for conv_id, messages in iterator:
        try:
            conv = tokenize_conversation(
                messages, tokenizer, conversation_id=conv_id, tools=tools
            )
            if max_length is not None and conv.seq_len > max_length:
                continue
            results.append(conv)
        except Exception as e:
            print(f"Warning: Failed to tokenize conversation {conv_id}: {e}")
            continue

    return results


# =============================================================================
# Analysis Results
# =============================================================================


@dataclass
class SectionMetrics:
    """Metrics aggregated for a single section."""

    role: str
    content_type: str
    token_count: int
    start_idx: int
    end_idx: int
    metrics: Dict[str, float]


@dataclass
class AnalysisResult:
    """Result of analyzing a single conversation."""

    conversation_id: str
    token_count: int
    sections: List[SectionMetrics]
    metrics_by_position: Dict[str, List[float]]
    overall_metrics: Dict[str, float]


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple conversations."""

    num_conversations: int
    total_tokens: int
    by_role: Dict[str, Dict[str, float]]
    by_content_type: Dict[str, Dict[str, float]]
    overall: Dict[str, float]
    by_position_normalized: Dict[str, List[float]]


# =============================================================================
# Backend Abstraction
# =============================================================================


def get_prefill_result_local(
    model,
    input_ids: torch.Tensor,
    device: Optional[torch.device] = None,
) -> PrefillResult:
    """Get PrefillResult from local HuggingFace model.

    Args:
        model: HuggingFace model.
        input_ids: Token IDs [seq_len].
        device: Device to run on (None lets model handle placement).

    Returns:
        PrefillResult with precomputed entropy (logits are not stored to save memory).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if device is not None:
        input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits.squeeze(0)
    input_ids = input_ids.squeeze(0)

    return hf_logits_to_prefill_result(logits, input_ids)


def get_prefill_result_fireworks(
    input_ids: torch.Tensor,
    model: str = "accounts/fireworks/models/gpt-oss-20b",
    api_key: Optional[str] = None,
) -> PrefillResult:
    """Get PrefillResult from Fireworks API.

    Args:
        input_ids: Token IDs [seq_len].
        model: Fireworks model name.
        api_key: API key (or from env).

    Returns:
        PrefillResult (without full logits).
    """
    token_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    return fireworks_prefill(token_ids, model=model, api_key=api_key)


def analyze_conversation(
    conversation: TokenizedConversation,
    backend: str = "local",
    model=None,
    device: Optional[torch.device] = None,
    fireworks_model: str = "accounts/fireworks/models/gpt-oss-20b",
    fireworks_api_key: Optional[str] = None,
    metrics: Optional[List[UnifiedMetric]] = None,
) -> AnalysisResult:
    """Analyze a single conversation using specified backend.

    Args:
        conversation: Tokenized conversation with section tracking.
        backend: "local" for HuggingFace model, "fireworks" for Fireworks API.
        model: HuggingFace model (required for local backend).
        device: Device for local computation.
        fireworks_model: Fireworks model name.
        fireworks_api_key: Fireworks API key.
        metrics: List of UnifiedMetric to compute. Defaults to standard set.

    Returns:
        AnalysisResult with per-token and per-section metrics.
    """
    # Get PrefillResult from appropriate backend
    if backend == "fireworks":
        prefill_result = get_prefill_result_fireworks(
            conversation.input_ids,
            model=fireworks_model,
            api_key=fireworks_api_key,
        )
    else:
        if model is None:
            raise ValueError("model is required for local backend")
        prefill_result = get_prefill_result_local(
            model, conversation.input_ids, device=device
        )

    # Use default metrics if not specified
    if metrics is None:
        metrics = [
            NegLogProbUnified(),
            PerplexityUnified(),
            EstimatedEntropyUnified(),  # Exact for HF, tight lower bound for API
            TopKEntropyUnified(),       # Renormalized top-k (comparable across backends)
            TopKMassUnified(),
        ]

    # Compute metrics using unified interface
    metric_values = compute_unified_metrics(prefill_result, metrics)
    token_count = prefill_result.seq_len

    # Per-section aggregation
    section_metrics = []
    for section in conversation.sections:
        section_data: Dict[str, float] = {}

        for metric_name, values in metric_values.items():
            start = section.start_idx
            end = min(len(values), section.end_idx)

            if end > start:
                section_values = values[start:end]
                valid_mask = ~torch.isnan(section_values)
                if valid_mask.any():
                    section_data[metric_name] = section_values[valid_mask].mean().item()
                else:
                    section_data[metric_name] = float("nan")
            else:
                section_data[metric_name] = float("nan")

        section_metrics.append(
            SectionMetrics(
                role=section.role,
                content_type=section.content_type,
                token_count=section.length,
                start_idx=section.start_idx,
                end_idx=section.end_idx,
                metrics=section_data,
            )
        )

    # Overall metrics (ignoring NaN)
    overall = {}
    metrics_by_position = {}
    for metric_name, values in metric_values.items():
        valid_mask = ~torch.isnan(values)
        if valid_mask.any():
            overall[metric_name] = values[valid_mask].mean().item()
        else:
            overall[metric_name] = float("nan")
        metrics_by_position[metric_name] = values.tolist()

    return AnalysisResult(
        conversation_id=conversation.conversation_id or "unknown",
        token_count=token_count,
        sections=section_metrics,
        metrics_by_position=metrics_by_position,
        overall_metrics=overall,
    )


def aggregate_results(results: List[AnalysisResult]) -> AggregatedMetrics:
    """Aggregate metrics across multiple conversations.

    Args:
        results: List of AnalysisResult objects.

    Returns:
        AggregatedMetrics with by-role and overall statistics.
    """
    if not results:
        return AggregatedMetrics(
            num_conversations=0,
            total_tokens=0,
            by_role={},
            by_content_type={},
            overall={},
            by_position_normalized={},
        )

    role_metrics: Dict[str, Dict[str, List[float]]] = {}
    content_type_metrics: Dict[str, Dict[str, List[float]]] = {}
    overall_values: Dict[str, List[float]] = {}

    total_tokens = 0
    for result in results:
        total_tokens += result.token_count

        # Overall
        for metric_name, value in result.overall_metrics.items():
            if metric_name not in overall_values:
                overall_values[metric_name] = []
            if not (isinstance(value, float) and math.isnan(value)):
                overall_values[metric_name].append(value)

        # By section
        for section in result.sections:
            role = section.role
            ctype = section.content_type

            if role not in role_metrics:
                role_metrics[role] = {}
            if ctype not in content_type_metrics:
                content_type_metrics[ctype] = {}

            for metric_name, value in section.metrics.items():
                if isinstance(value, float) and math.isnan(value):
                    continue
                if metric_name not in role_metrics[role]:
                    role_metrics[role][metric_name] = []
                role_metrics[role][metric_name].append(value)

                if metric_name not in content_type_metrics[ctype]:
                    content_type_metrics[ctype][metric_name] = []
                content_type_metrics[ctype][metric_name].append(value)

    # Compute means
    by_role = {}
    for role, metrics in role_metrics.items():
        by_role[role] = {
            name: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }

    by_content_type = {}
    for ctype, metrics in content_type_metrics.items():
        by_content_type[ctype] = {
            name: sum(values) / len(values)
            for name, values in metrics.items()
            if values
        }

    overall = {
        name: sum(values) / len(values)
        for name, values in overall_values.items()
        if values
    }

    return AggregatedMetrics(
        num_conversations=len(results),
        total_tokens=total_tokens,
        by_role=by_role,
        by_content_type=by_content_type,
        overall=overall,
        by_position_normalized={},
    )


# =============================================================================
# Plot Source Data Export
# =============================================================================


def _open_text_file(path: Path) -> IO[str]:
    """Open a text file for writing, supporting optional gzip via .gz suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def save_token_plot_data_csv(
    results: List[Any],
    metrics: List[TokenMetric],
    output_path: Path,
):
    """Save token-level plot source data to a CSV for fast re-plotting.

    Writes one row per token with role/content_type labels derived from section
    boundaries and one column per metric. Shifted metrics are aligned to the
    predicted token position (i+1).

    Args:
        results: List of AnalysisResult-like objects (must include per-token metric arrays).
        metrics: Metric objects (used for name + is_shifted alignment).
        output_path: Output CSV path. Use ".csv.gz" to gzip.
    """
    metric_names = [m.name for m in metrics]
    metric_lookup = {m.name: m for m in metrics}

    fieldnames = [
        "conversation_id",
        "token_idx",
        "section_idx",
        "role",
        "content_type",
        *metric_names,
    ]

    with _open_text_file(output_path) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            token_count = int(getattr(result, "token_count"))

            roles = ["unknown"] * token_count
            content_types = ["unknown"] * token_count
            section_indices = [-1] * token_count
            for idx, section in enumerate(getattr(result, "sections")):
                start = max(0, int(getattr(section, "start_idx")))
                end = min(token_count, int(getattr(section, "end_idx")))
                if end <= start:
                    continue
                role = str(getattr(section, "role"))
                content_type = str(getattr(section, "content_type"))
                roles[start:end] = [role] * (end - start)
                content_types[start:end] = [content_type] * (end - start)
                section_indices[start:end] = [idx] * (end - start)

            metrics_by_position = getattr(result, "metrics_by_position", {}) or {}

            # Align all metrics to token positions (0..token_count-1).
            aligned_by_metric: Dict[str, List[float]] = {}
            for name in metric_names:
                values = metrics_by_position.get(name)
                aligned = [float("nan")] * token_count

                if values is None:
                    aligned_by_metric[name] = aligned
                    continue

                is_shifted = metric_lookup.get(name).is_shifted if name in metric_lookup else False
                if is_shifted and len(values) != token_count:
                    # Shifted metrics usually have length seq_len-1; index i predicts token i+1.
                    for i, v in enumerate(values):
                        pos = i + 1
                        if pos >= token_count:
                            break
                        aligned[pos] = v
                else:
                    for i, v in enumerate(values[:token_count]):
                        aligned[i] = v

                aligned_by_metric[name] = aligned

            for token_idx in range(token_count):
                row: Dict[str, Any] = {
                    "conversation_id": getattr(result, "conversation_id", "unknown"),
                    "token_idx": token_idx,
                    "section_idx": section_indices[token_idx],
                    "role": roles[token_idx],
                    "content_type": content_types[token_idx],
                }
                for name in metric_names:
                    v = aligned_by_metric[name][token_idx]
                    row[name] = v if math.isfinite(v) else ""
                writer.writerow(row)

    print(f"Saved token-level plot data to {output_path}")
