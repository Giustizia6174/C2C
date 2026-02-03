"""
SGLang support for perplexity analysis (archived).

This module contains SGLang-related code moved from rosetta/utils/core.py
and script/workflow/analysis/perplexity.py for archival purposes.

The SGLang backend is no longer actively used in the main analysis pipeline.
"""

from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import torch

from rosetta.workflow.analysis.interface import (
    TokenizedConversation,
    extract_conversations,
    load_evaluation_results,
)
from rosetta.workflow.analysis.oss_tokenizer import batch_tokenize_with_sections


# =============================================================================
# SGLang Prefill API
# =============================================================================


@dataclass
class SGLangPrefillResult:
    """Result from SGLang prefill operation.

    Attributes:
        input_token_logprobs: Log probability of each input token.
        input_top_logprobs: Top-k logprobs for each input position.
            Each element is a list of [logprob, token_id, token_text] tuples.
        output_token_logprobs: Log probability of each generated token (if any).
        output_top_logprobs: Top-k logprobs for each output position.
        generated_text: Generated text (if max_new_tokens > 0).
        meta_info: Full meta_info dict from SGLang response.
    """

    input_token_logprobs: List[float]
    input_top_logprobs: List[List[Any]]
    output_token_logprobs: Optional[List[float]] = None
    output_top_logprobs: Optional[List[List[Any]]] = None
    generated_text: Optional[str] = None
    meta_info: Optional[Dict[str, Any]] = None


def _normalize_sglang_response(resp_json: Union[List, Dict]) -> Dict:
    """Normalize SGLang response to a single dict."""
    if isinstance(resp_json, list):
        if not resp_json:
            raise RuntimeError("Empty response list from /generate")
        resp_json = resp_json[0]
    if not isinstance(resp_json, dict):
        raise TypeError(f"Unexpected /generate response type: {type(resp_json)!r}")
    if "error" in resp_json:
        raise RuntimeError(resp_json["error"])
    return resp_json


def sglang_prefill(
    text: str,
    base_url: str = "http://127.0.0.1:30000",
    top_logprobs_num: int = 20,
    max_new_tokens: int = 0,
    temperature: float = 0.0,
    timeout: int = 120,
) -> SGLangPrefillResult:
    """Prefill text through SGLang server and get token-level statistics.

    This calls the SGLang /generate endpoint with logprob_start_len=0 to get
    logprobs for all input tokens (prompt scoring mode).

    Args:
        text: Input text to prefill.
        base_url: SGLang server URL (default: http://127.0.0.1:30000).
        top_logprobs_num: Number of top logprobs to return per position (default: 20).
        max_new_tokens: Max tokens to generate. Use 0 for prefill-only (default: 0).
        temperature: Sampling temperature (default: 0.0 for greedy).
        timeout: Request timeout in seconds (default: 120).

    Returns:
        SGLangPrefillResult with token logprobs and top-k distributions.

    Raises:
        RuntimeError: If server returns an error.
        requests.RequestException: On HTTP errors.
    """
    base_url = base_url.rstrip("/")
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "top_logprobs_num": top_logprobs_num,
        "return_text_in_logprobs": True,
        "logprob_start_len": 0,  # Return logprobs for all input tokens
    }

    resp = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    resp_json = _normalize_sglang_response(resp.json())

    meta_info = resp_json.get("meta_info") or {}
    input_token_logprobs = meta_info.get("input_token_logprobs")
    input_top_logprobs = meta_info.get("input_top_logprobs")

    if input_token_logprobs is None:
        raise RuntimeError(
            "Server response did not include meta_info.input_token_logprobs. "
            "Ensure return_logprob=true and logprob_start_len=0 in the request."
        )

    return SGLangPrefillResult(
        input_token_logprobs=input_token_logprobs,
        input_top_logprobs=input_top_logprobs,
        output_token_logprobs=meta_info.get("output_token_logprobs"),
        output_top_logprobs=meta_info.get("output_top_logprobs"),
        generated_text=resp_json.get("text"),
        meta_info=meta_info,
    )


# =============================================================================
# Entropy Computation from Top-K Logprobs
# =============================================================================


def _entropy_from_top_logprobs(top_logprobs: List[Any]) -> Optional[float]:
    """Compute Shannon entropy from renormalized top-k distribution.

    This renormalizes the top-k probabilities to sum to 1.0, then computes
    entropy. This is an approximation since we don't have the full distribution.

    Args:
        top_logprobs: Top-k logprobs as [[logprob, token_id, token_text], ...].

    Returns:
        Entropy in nats, or None if input is empty.
    """
    if not top_logprobs:
        return None

    logps = []
    for item in top_logprobs:
        if isinstance(item, (list, tuple)) and item:
            logps.append(float(item[0]))
        elif isinstance(item, dict) and "logprob" in item:
            logps.append(float(item["logprob"]))
        else:
            raise TypeError(f"Unexpected top_logprobs item: {item!r}")

    # Renormalize using log-sum-exp trick for numerical stability
    max_logp = max(logps)
    probs = [math.exp(lp - max_logp) for lp in logps]
    z = sum(probs)
    probs = [p / z for p in probs]

    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _entropy_lower_bound_from_top_logprobs(
    top_logprobs: List[Any],
) -> tuple[Optional[float], Optional[float]]:
    """Compute lower bound on entropy using top-k mass + "other" bucket.

    Uses absolute token probabilities from logprobs:
    - p_i = exp(logprob_i) for i in top-k
    - r = 1 - sum_i p_i (remaining tail mass)

    Then computes H([p_1..p_k, r]). The true entropy is >= this value
    because splitting the tail into many tokens can only increase entropy.

    Args:
        top_logprobs: Top-k logprobs as [[logprob, token_id, token_text], ...].

    Returns:
        Tuple of (entropy_lower_bound, top_k_mass), or (None, None) if empty.
    """
    if not top_logprobs:
        return None, None

    probs = []
    for item in top_logprobs:
        if isinstance(item, (list, tuple)) and item:
            probs.append(math.exp(float(item[0])))
        elif isinstance(item, dict) and "logprob" in item:
            probs.append(math.exp(float(item["logprob"])))
        else:
            raise TypeError(f"Unexpected top_logprobs item: {item!r}")

    topk_mass = sum(probs)
    tail_mass = max(0.0, 1.0 - topk_mass)  # Guard against rounding > 1.0

    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    if tail_mass > 0.0:
        entropy -= tail_mass * math.log(tail_mass)

    return entropy, topk_mass


def _extract_logprob(item: Any) -> Optional[float]:
    """Extract log probability from SGLang token logprob item.

    SGLang returns token logprobs as [logprob, token_id, token_text] tuples.
    The first token typically has None as logprob (no context to predict from).

    Args:
        item: A [logprob, token_id, token_text] tuple or None.

    Returns:
        The log probability as a float, or None if not available.
    """
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 1:
        lp = item[0]
        return float(lp) if lp is not None else None
    if isinstance(item, dict) and "logprob" in item:
        lp = item["logprob"]
        return float(lp) if lp is not None else None
    return None


# =============================================================================
# Metrics Computation from SGLang Response
# =============================================================================


def compute_metrics_from_sglang(
    result: SGLangPrefillResult,
    include_output: bool = False,
) -> Dict[str, List[Optional[float]]]:
    """Compute metrics from SGLang prefill result.

    Args:
        result: SGLangPrefillResult from sglang_prefill().
        include_output: If True, also compute metrics for generated tokens.

    Returns:
        Dict with per-token metrics:
        - "neg_log_prob": Negative log probability of each token.
        - "entropy_topk": Entropy from renormalized top-k distribution (approximation).
        - "entropy_lower": Lower bound on true entropy using top-k + tail bucket.
        - "topk_mass": Mass covered by top-k tokens (indicates approximation quality).
        If include_output=True, also includes "output_*" versions.

        Note: First token typically has None values (no prior context).
    """
    metrics: Dict[str, List[Optional[float]]] = {}

    # Input token metrics
    # SGLang returns [logprob, token_id, token_text] tuples
    if result.input_token_logprobs:
        neg_log_probs = []
        for item in result.input_token_logprobs:
            lp = _extract_logprob(item)
            neg_log_probs.append(-lp if lp is not None else None)
        metrics["neg_log_prob"] = neg_log_probs

    if result.input_top_logprobs:
        entropies_topk = []
        entropies_lower = []
        topk_masses = []

        for tok_topk in result.input_top_logprobs:
            if tok_topk is None:
                entropies_topk.append(None)
                entropies_lower.append(None)
                topk_masses.append(None)
            else:
                entropies_topk.append(_entropy_from_top_logprobs(tok_topk))
                h_lower, mass = _entropy_lower_bound_from_top_logprobs(tok_topk)
                entropies_lower.append(h_lower)
                topk_masses.append(mass)

        metrics["entropy_topk"] = entropies_topk
        metrics["entropy_lower"] = entropies_lower
        metrics["topk_mass"] = topk_masses

    # Output token metrics (if requested and available)
    if include_output:
        if result.output_token_logprobs:
            out_neg_log_probs = []
            for item in result.output_token_logprobs:
                lp = _extract_logprob(item)
                out_neg_log_probs.append(-lp if lp is not None else None)
            metrics["output_neg_log_prob"] = out_neg_log_probs

        if result.output_top_logprobs:
            out_entropies_topk = []
            out_entropies_lower = []
            out_topk_masses = []

            for tok_topk in result.output_top_logprobs:
                if tok_topk is None:
                    out_entropies_topk.append(None)
                    out_entropies_lower.append(None)
                    out_topk_masses.append(None)
                else:
                    out_entropies_topk.append(_entropy_from_top_logprobs(tok_topk))
                    h_lower, mass = _entropy_lower_bound_from_top_logprobs(tok_topk)
                    out_entropies_lower.append(h_lower)
                    out_topk_masses.append(mass)

            metrics["output_entropy_topk"] = out_entropies_topk
            metrics["output_entropy_lower"] = out_entropies_lower
            metrics["output_topk_mass"] = out_topk_masses

    return metrics


def sglang_prefill_and_compute_metrics(
    text: str,
    base_url: str = "http://127.0.0.1:30000",
    top_logprobs_num: int = 20,
    max_new_tokens: int = 0,
    temperature: float = 0.0,
    timeout: int = 120,
    include_output: bool = False,
) -> Dict[str, List[float]]:
    """Prefill text through SGLang and compute token-level metrics.

    Convenience function that combines sglang_prefill() and compute_metrics_from_sglang().

    Args:
        text: Input text to prefill.
        base_url: SGLang server URL.
        top_logprobs_num: Number of top logprobs to return per position.
        max_new_tokens: Max tokens to generate (0 for prefill-only).
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        include_output: If True, also compute metrics for generated tokens.

    Returns:
        Dict with per-token metrics (see compute_metrics_from_sglang).
    """
    result = sglang_prefill(
        text=text,
        base_url=base_url,
        top_logprobs_num=top_logprobs_num,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return compute_metrics_from_sglang(result, include_output=include_output)


# =============================================================================
# SGLang Perplexity Analyzer
# =============================================================================


# Import AnalysisResult and AggregatedMetrics types for type hints
# These are defined in perplexity.py but we need them here
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


class SGLangPerplexityAnalyzer:
    """Perplexity/entropy analyzer using SGLang HTTP API.

    This analyzer uses an SGLang server instead of loading the model locally.
    It computes metrics via the /generate endpoint with logprob_start_len=0.

    Key differences from HuggingFace PerplexityAnalyzer:
    - entropy_topk: Renormalized top-k entropy (approximation, not full vocab)
    - entropy_lower: Lower bound on true entropy using top-k + tail bucket
    - topk_mass: Probability mass covered by top-k tokens
    - First token has None values (no prior context to predict from)
    """

    def __init__(
        self,
        tokenizer,
        base_url: str = "http://127.0.0.1:30000",
        top_logprobs_num: int = 10,
        timeout: int = 120,
    ):
        """Initialize the SGLang analyzer.

        Args:
            tokenizer: Tokenizer for the model (used for section mapping).
            base_url: SGLang server URL.
            top_logprobs_num: Number of top logprobs to request per position.
            timeout: Request timeout in seconds.
        """
        self.tokenizer = tokenizer
        self.base_url = base_url
        self.top_logprobs_num = top_logprobs_num
        self.timeout = timeout

    def analyze_conversation(
        self,
        conversation: TokenizedConversation,
        align_mode: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze a single tokenized conversation via SGLang.

        Args:
            conversation: Tokenized conversation with section tracking.
            align_mode: Not used for SGLang (metrics already aligned).

        Returns:
            AnalysisResult with per-token and per-section metrics.
        """
        # Decode tokens back to text for SGLang
        text = self.tokenizer.decode(conversation.input_ids)

        # Call SGLang server
        result = sglang_prefill(
            text=text,
            base_url=self.base_url,
            top_logprobs_num=self.top_logprobs_num,
            max_new_tokens=0,
            timeout=self.timeout,
        )

        # Compute metrics from SGLang response
        sglang_metrics = compute_metrics_from_sglang(result, include_output=False)

        # Convert to tensors, handling None values
        metric_values: Dict[str, torch.Tensor] = {}
        for name, values in sglang_metrics.items():
            # Replace None with NaN for tensor conversion
            float_values = [v if v is not None else float("nan") for v in values]
            metric_values[name] = torch.tensor(float_values)

        # Per-section aggregation
        section_metrics = []
        for section in conversation.sections:
            section_data = {"role": section.role, "content_type": section.content_type}
            section_data["token_count"] = section.length
            section_data["metrics"] = {}

            for metric_name, values in metric_values.items():
                # SGLang metrics are shifted: value[i] is for token i
                # (predicting token i given context 0:i-1)
                start = section.start_idx
                end = min(len(values), section.end_idx)

                if end > start:
                    section_values = values[start:end]
                    # Compute mean, ignoring NaN values
                    valid_mask = ~torch.isnan(section_values)
                    if valid_mask.any():
                        section_data["metrics"][metric_name] = (
                            section_values[valid_mask].mean().item()
                        )
                    else:
                        section_data["metrics"][metric_name] = float("nan")
                else:
                    section_data["metrics"][metric_name] = float("nan")

            section_metrics.append(
                SectionMetrics(
                    role=section.role,
                    content_type=section.content_type,
                    token_count=section.length,
                    start_idx=section.start_idx,
                    end_idx=section.end_idx,
                    metrics=section_data["metrics"],
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
            token_count=conversation.seq_len,
            sections=section_metrics,
            metrics_by_position=metrics_by_position,
            overall_metrics=overall,
        )

    def analyze_file(
        self,
        path: Path,
        limit: Optional[int] = None,
        max_length: Optional[int] = None,
        show_progress: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        exclude_final: bool = True,
    ) -> List[AnalysisResult]:
        """Analyze all conversations in a JSONL file via SGLang.

        Args:
            path: Path to the evaluation JSONL file.
            limit: Maximum number of conversations to analyze.
            max_length: Maximum sequence length (skip longer).
            show_progress: Whether to show progress bar.
            tools: Optional list of tool schemas for section detection.
            exclude_final: If True, exclude final message to include all reasoning.

        Returns:
            List of AnalysisResult objects.
        """
        # Load and extract conversations
        records = load_evaluation_results(path)
        if limit:
            records = records[:limit]

        conversations = extract_conversations(records)

        # Use section-aware tokenizer for gpt-oss models
        tokenized = batch_tokenize_with_sections(
            conversations,
            self.tokenizer,
            tools=tools,
            max_length=max_length,
            show_progress=show_progress,
            exclude_final=exclude_final,
            convert_reasoning=True,
        )

        # Analyze each conversation
        results = []
        iterator = tokenized
        if show_progress:
            try:
                from rich.progress import track

                iterator = track(tokenized, description="Analyzing (SGLang)...")
            except ImportError:
                pass

        for conv in iterator:
            try:
                result = self.analyze_conversation(conv)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to analyze {conv.conversation_id}: {e}")
                continue

        return results

    def aggregate(self, results: List[AnalysisResult]) -> AggregatedMetrics:
        """Aggregate metrics across multiple conversations.

        Uses the same aggregation logic as PerplexityAnalyzer.
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

        # Collect by role
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
                if not (isinstance(value, float) and value != value):  # Check for NaN
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
                    if isinstance(value, float) and value != value:  # Skip NaN
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
                name: sum(values) / len(values) for name, values in metrics.items() if values
            }

        by_content_type = {}
        for ctype, metrics in content_type_metrics.items():
            by_content_type[ctype] = {
                name: sum(values) / len(values) for name, values in metrics.items() if values
            }

        overall = {
            name: sum(values) / len(values) for name, values in overall_values.items() if values
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
# SGLang CSV Output
# =============================================================================


def save_sglang_plot_data_csv(
    results: List[AnalysisResult],
    output_path: Path,
):
    """Save SGLang token-level plot data to CSV.

    SGLang metrics (neg_log_prob, entropy_topk, entropy_lower, topk_mass) are
    not shifted in the same way as HF metrics - they are aligned to the token
    position directly.

    Args:
        results: List of AnalysisResult objects from SGLangPerplexityAnalyzer.
        output_path: Output CSV path. Use ".csv.gz" to gzip.
    """
    # Collect all metric names from results
    metric_names = set()
    for result in results:
        metric_names.update(result.metrics_by_position.keys())
    metric_names = sorted(metric_names)

    fieldnames = [
        "conversation_id",
        "token_idx",
        "section_idx",
        "role",
        "content_type",
        *metric_names,
    ]

    # Determine if gzip
    if str(output_path).endswith(".gz"):
        open_fn = lambda p: gzip.open(p, "wt", encoding="utf-8")
    else:
        open_fn = lambda p: open(p, "w", encoding="utf-8", newline="")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_fn(output_path) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            token_count = result.token_count

            # Build role/content_type arrays from sections
            roles = ["unknown"] * token_count
            content_types = ["unknown"] * token_count
            section_indices = [-1] * token_count
            for idx, section in enumerate(result.sections):
                start = max(0, section.start_idx)
                end = min(token_count, section.end_idx)
                if end <= start:
                    continue
                roles[start:end] = [section.role] * (end - start)
                content_types[start:end] = [section.content_type] * (end - start)
                section_indices[start:end] = [idx] * (end - start)

            # Get metrics by position (SGLang metrics are not shifted)
            metrics_by_position = result.metrics_by_position or {}

            for token_idx in range(token_count):
                row = {
                    "conversation_id": result.conversation_id,
                    "token_idx": token_idx,
                    "section_idx": section_indices[token_idx],
                    "role": roles[token_idx],
                    "content_type": content_types[token_idx],
                }
                for name in metric_names:
                    values = metrics_by_position.get(name, [])
                    if token_idx < len(values):
                        row[name] = values[token_idx]
                    else:
                        row[name] = float("nan")
                writer.writerow(row)

    print(f"Saved SGLang plot data to {output_path}")
