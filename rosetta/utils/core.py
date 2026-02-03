"""
Core utilities for Cache-to-Cache (C2C) operations.

Includes:
- Sharer/mask conversion utilities for KV-cache projection
- Token-level metric computations (entropy, perplexity, etc.)
- Model prefill utilities for analysis
- Fireworks API integration for remote logprob computation
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def sharers_to_mask(sharer_indices: List[int]) -> int:
    """
    Convert a list of sharer indices to a bitmask.
    
    Args:
        sharer_indices: List of 1-based sharer indices (e.g., [1, 2, 3])
        
    Returns:
        Bitmask integer (e.g., [1, 2] -> 3, [1, 3] -> 5, [1, 2, 3] -> 7)
    
    Example:
        >>> sharers_to_mask([1])      # 001 = 1
        1
        >>> sharers_to_mask([2])      # 010 = 2
        2
        >>> sharers_to_mask([1, 2])   # 011 = 3
        3
        >>> sharers_to_mask([1, 3])   # 101 = 5
        5
    """
    mask = 0
    for idx in sharer_indices:
        mask |= (1 << (idx - 1))
    return mask


def mask_to_sharers(mask: int) -> List[int]:
    """
    Convert a bitmask to a list of sharer indices.
    
    Args:
        mask: Bitmask integer
        
    Returns:
        List of 1-based sharer indices
    
    Example:
        >>> mask_to_sharers(1)   # 001 -> [1]
        [1]
        >>> mask_to_sharers(3)   # 011 -> [1, 2]
        [1, 2]
        >>> mask_to_sharers(5)   # 101 -> [1, 3]
        [1, 3]
        >>> mask_to_sharers(7)   # 111 -> [1, 2, 3]
        [1, 2, 3]
    """
    if mask <= 0:
        return []
    sharers = []
    idx = 1
    while mask:
        if mask & 1:
            sharers.append(idx)
        mask >>= 1
        idx += 1
    return sharers


def all_sharers_mask(num_sharers: int) -> int:
    """
    Get bitmask that selects all sharers.
    
    Args:
        num_sharers: Number of sharers
        
    Returns:
        Bitmask with all bits set (e.g., 3 sharers -> 7 = 111)
    """
    return (1 << num_sharers) - 1


def format_sharer_mask(mask: int) -> str:
    """
    Format a sharer mask as a human-readable string.
    
    Args:
        mask: Bitmask integer (-1=no projection, 0=self projection, >0=sharer bitmask)
        
    Returns:
        Formatted string like "sharers [1, 2]" or "no projection"
    """
    if mask < 0:
        return "no projection"
    if mask == 0:
        return "self projection"
    sharers = mask_to_sharers(mask)
    return f"sharers {sharers}"


# =============================================================================
# Token-Level Metrics
# =============================================================================


class TokenMetric(ABC):
    """Abstract base class for per-token metrics computed from model outputs.

    Subclasses implement specific metrics like entropy, perplexity, etc.
    All metrics take logits and input_ids, returning per-token values.

    Metrics can be "shifted" or "non-shifted":
    - Non-shifted (is_shifted=False): Output length equals input length (seq_len).
      Example: entropy[i] = uncertainty at position i.
    - Shifted (is_shifted=True): Output length is seq_len-1.
      Example: perplexity[i] = how surprised we are by token at position i+1.
      The metric at index i corresponds to predicting token i+1 from context 0:i.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name for reporting and identification."""
        ...

    @property
    def is_shifted(self) -> bool:
        """Whether this metric is shifted for next-token prediction.

        If True, the metric has length seq_len-1 and value[i] corresponds
        to predicting the token at position i+1. Used for alignment in
        visualization and analysis.

        Default: False (same length as input sequence).
        """
        return False

    @abstractmethod
    def compute(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute metric for each token position.

        Args:
            logits: Model output logits [seq_len, vocab_size] or [batch, seq_len, vocab_size].
            input_ids: Input token IDs [seq_len] or [batch, seq_len].

        Returns:
            Per-token metric values with same batch dimensions as input.
            Length is seq_len if is_shifted=False, seq_len-1 if is_shifted=True.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class EntropyMetric(TokenMetric):
    """Per-token entropy of the probability distribution.

    Entropy H(p) = -sum(p * log(p)) measures the uncertainty in the
    model's predictions. Higher entropy means more uncertainty.
    """

    @property
    def name(self) -> str:
        return "entropy"

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # H(p) = -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Avoid -inf * 0 = nan by using where
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class PerplexityMetric(TokenMetric):
    """Per-token perplexity (exp of cross-entropy loss).

    For each position i, computes exp(-log(p(token_{i+1} | token_{0:i}))).
    This is the perplexity of predicting the next token.

    Note: Returns values for positions 0 to seq_len-2 (predicting tokens 1 to seq_len-1).
    The last position has no next token to predict.
    """

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Shift for next-token prediction
        # logits[i] predicts token at position i+1
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        # Cross-entropy loss per token
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)

        # Perplexity = exp(CE)
        return torch.exp(ce_loss)


class NegLogProbMetric(TokenMetric):
    """Negative log probability of each token.

    For position i, returns -log(p(token_{i+1} | token_{0:i})).
    Lower values indicate the model is more confident about the token.

    Note: Returns values for positions 0 to seq_len-2.
    """

    @property
    def name(self) -> str:
        return "neg_log_prob"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return -token_log_probs


class TopKAccuracyMetric(TokenMetric):
    """Whether the true next token is in the top-k predictions.

    Returns 1.0 if the actual next token is among the top-k predictions,
    0.0 otherwise. Useful for measuring prediction quality.
    """

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"top{self.k}_accuracy"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        # Get top-k predictions
        _, top_k_indices = torch.topk(shift_logits, k=self.k, dim=-1)

        # Check if true label is in top-k
        labels_expanded = shift_labels.unsqueeze(-1).expand_as(top_k_indices)
        in_top_k = (top_k_indices == labels_expanded).any(dim=-1)

        return in_top_k.float()


class RankMetric(TokenMetric):
    """Rank of the true next token in the probability distribution.

    Returns the 1-based rank of the actual next token. Rank 1 means
    the model's top prediction was correct.
    """

    @property
    def name(self) -> str:
        return "rank"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        # Sort indices by logit value (descending)
        sorted_indices = torch.argsort(shift_logits, dim=-1, descending=True)

        # Find rank of each true label
        # Create position indices
        ranks = torch.zeros_like(shift_labels, dtype=torch.float)
        for i in range(shift_logits.size(-1)):
            matches = sorted_indices[..., i] == shift_labels
            ranks[matches] = i + 1  # 1-based rank

        return ranks


# Default metrics for convenience
DEFAULT_METRICS = [EntropyMetric(), PerplexityMetric(), NegLogProbMetric()]


def get_metric_by_name(name: str) -> TokenMetric:
    """Get a metric instance by name.

    Args:
        name: Metric name ("entropy", "perplexity", "neg_log_prob", "rank", "topN_accuracy").

    Returns:
        TokenMetric instance.
    """
    name = name.lower()
    if name == "entropy":
        return EntropyMetric()
    elif name == "perplexity":
        return PerplexityMetric()
    elif name == "neg_log_prob":
        return NegLogProbMetric()
    elif name == "rank":
        return RankMetric()
    elif name.startswith("top") and name.endswith("_accuracy"):
        k = int(name[3:-9])
        return TopKAccuracyMetric(k=k)
    else:
        raise ValueError(f"Unknown metric: {name}")


# =============================================================================
# Model Prefill and Metric Computation
# =============================================================================


def prefill_and_compute_metrics(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    metrics: Optional[List[TokenMetric]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Prefill model with input sequence and compute metrics.

    This runs a forward pass through the model to get logits for all
    positions, then computes the requested metrics.

    Args:
        model: HuggingFace model (or RosettaModel).
        input_ids: Token IDs [seq_len] or [batch, seq_len].
        metrics: List of metric objects to compute. Defaults to entropy, perplexity, neg_log_prob.
        attention_mask: Optional attention mask.
        device: Device to move inputs to. If None, lets the model handle device
            placement (works with device_map="auto" for multi-GPU setups).

    Returns:
        Dict mapping metric name to per-token values tensor.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # Ensure batch dimension
    was_1d = input_ids.dim() == 1
    if was_1d:
        input_ids = input_ids.unsqueeze(0)

    # Only move to device if explicitly specified
    # For device_map="auto", the model handles placement internally
    if device is not None:
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    logits = outputs.logits
    if was_1d:
        logits = logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    # Compute metrics
    results = {}
    for metric in metrics:
        values = metric.compute(logits, input_ids)
        results[metric.name] = values

    return results


def compute_metrics_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    metrics: Optional[List[TokenMetric]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute metrics from pre-computed logits.

    Useful when you already have logits from a forward pass.

    Args:
        logits: Model output logits [seq_len, vocab_size] or [batch, seq_len, vocab_size].
        input_ids: Input token IDs [seq_len] or [batch, seq_len].
        metrics: List of metric objects. Defaults to entropy, perplexity, neg_log_prob.

    Returns:
        Dict mapping metric name to per-token values tensor.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    return {m.name: m.compute(logits, input_ids) for m in metrics}


def align_metrics(
    metrics_dict: Dict[str, torch.Tensor],
    metric_objects: Optional[List[TokenMetric]] = None,
    mode: str = "truncate",
) -> Dict[str, torch.Tensor]:
    """Align all metrics to the same length.

    Different metrics may have different lengths:
    - Non-shifted metrics (entropy): length seq_len
    - Shifted metrics (perplexity, neg_log_prob): length seq_len-1

    This function aligns them to a common length for easier comparison.

    Args:
        metrics_dict: Dict mapping metric name to per-token values tensor.
        metric_objects: Optional list of TokenMetric objects to get is_shifted info.
            If not provided, infers from tensor lengths (shortest is assumed shifted).
        mode: Alignment mode:
            - "truncate": Truncate longer metrics to match shorter ones (removes first token).
            - "pad_nan": Pad shorter metrics with NaN at position 0.
            - "pad_zero": Pad shorter metrics with 0 at position 0.

    Returns:
        Dict with all metrics aligned to the same length.
    """
    if not metrics_dict:
        return metrics_dict

    # Get lengths
    lengths = {name: len(values) for name, values in metrics_dict.items()}
    min_len = min(lengths.values())
    max_len = max(lengths.values())

    if min_len == max_len:
        return metrics_dict  # Already aligned

    # Determine which metrics are shifted
    if metric_objects is not None:
        shifted = {m.name: m.is_shifted for m in metric_objects if m.name in metrics_dict}
    else:
        # Infer: shorter length = shifted
        shifted = {name: length == min_len for name, length in lengths.items()}

    aligned = {}
    for name, values in metrics_dict.items():
        if mode == "truncate":
            # Truncate non-shifted metrics from the start (remove position 0)
            if not shifted.get(name, False) and len(values) > min_len:
                aligned[name] = values[1:]  # Remove first element
            else:
                aligned[name] = values
        elif mode == "pad_nan":
            # Pad shifted metrics with NaN at the start
            if shifted.get(name, True) and len(values) < max_len:
                pad = torch.full((1,), float("nan"), dtype=values.dtype, device=values.device)
                aligned[name] = torch.cat([pad, values])
            else:
                aligned[name] = values
        elif mode == "pad_zero":
            # Pad shifted metrics with 0 at the start
            if shifted.get(name, True) and len(values) < max_len:
                pad = torch.zeros((1,), dtype=values.dtype, device=values.device)
                aligned[name] = torch.cat([pad, values])
            else:
                aligned[name] = values
        else:
            raise ValueError(f"Unknown alignment mode: {mode}")

    return aligned


def prefill_and_compute_metrics_aligned(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    metrics: Optional[List[TokenMetric]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    align_mode: str = "truncate",
) -> Dict[str, torch.Tensor]:
    """Prefill model and compute aligned metrics.

    Convenience function that computes metrics and aligns them to the same length.

    Args:
        model: HuggingFace model.
        input_ids: Token IDs [seq_len] or [batch, seq_len].
        metrics: List of metric objects. Defaults to entropy, perplexity, neg_log_prob.
        attention_mask: Optional attention mask.
        device: Device to run on.
        align_mode: Alignment mode ("truncate", "pad_nan", "pad_zero").

    Returns:
        Dict with aligned metrics (all same length).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    results = prefill_and_compute_metrics(
        model=model,
        input_ids=input_ids,
        metrics=metrics,
        attention_mask=attention_mask,
        device=device,
    )

    return align_metrics(results, metric_objects=metrics, mode=align_mode)


# =============================================================================
# Unified Prefill Result Interface
# =============================================================================


@dataclass
class PrefillResult:
    """Unified result format from model prefill (local or API).

    This provides a common interface for metric computation regardless of
    whether logprobs come from a local HuggingFace model or a remote API.

    Attributes:
        seq_len: Number of tokens in the sequence.
        token_logprobs: Log probability of each actual token [seq_len].
            First token may be None/NaN (no prior context).
        top_k_logprobs: Top-k alternatives per position.
            Each element is a list of (token_id, logprob) tuples.
        input_ids: Original input token IDs [seq_len].
        precomputed_entropy: Pre-computed exact entropy [seq_len] for local models.
            Aligned to token positions (entropy[i] is for predicting token_i).
            None for API backends (uses approximation from top-k).
    """

    seq_len: int
    token_logprobs: torch.Tensor  # [seq_len]
    top_k_logprobs: List[List[tuple]]  # List of [(token_id, logprob), ...]
    input_ids: torch.Tensor  # [seq_len]
    precomputed_entropy: Optional[torch.Tensor] = None  # [seq_len]

    @property
    def has_exact_entropy(self) -> bool:
        """Whether exact entropy is available (precomputed from local model)."""
        return self.precomputed_entropy is not None


def hf_logits_to_prefill_result(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int = 5,
) -> PrefillResult:
    """Convert HuggingFace model logits to unified PrefillResult.

    In causal LM, logits[i] predicts the token at position i+1:
      - P(token_{i+1} | token_{0:i}) = softmax(logits[i])

    To get the logprob of each token given its preceding context:
      - P(token_0 | <nothing>) = undefined (use NaN)
      - P(token_i | token_{0:i-1}) = softmax(logits[i-1])[token_i] for i > 0

    This matches the Fireworks API convention where position 0 has no logprob
    (or logprob=0 as placeholder) since there's no prior context.

    Computes exact entropy from logits before discarding them to save memory.

    Args:
        logits: Model output logits [seq_len, vocab_size] or [batch, seq_len, vocab_size].
        input_ids: Input token IDs [seq_len] or [batch, seq_len].
        top_k: Number of top logprobs to extract.

    Returns:
        PrefillResult with precomputed entropy (logits are not stored).
    """
    # Handle batch dimension
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)

    seq_len = logits.size(0)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Extract logprob of actual tokens with CORRECT SHIFT
    # logits[i] predicts token[i+1], so to get P(token[i] | context):
    # - Position 0: NaN (no prior context to predict from)
    # - Position i (i > 0): log_probs[i-1, input_ids[i]]
    shifted_logprobs = log_probs[:-1].gather(-1, input_ids[1:].unsqueeze(-1)).squeeze(-1)
    first_token_logprob = torch.tensor([float("nan")], dtype=shifted_logprobs.dtype, device=shifted_logprobs.device)
    token_logprobs = torch.cat([first_token_logprob, shifted_logprobs])

    # Extract top-k logprobs per position with CORRECT SHIFT
    # top_k at position i should be alternatives for predicting token_i,
    # which comes from the distribution at position i-1.
    # - Position 0: empty (no prior distribution)
    # - Position i (i > 0): top-k from log_probs[i-1]
    top_k_values, top_k_indices = torch.topk(log_probs, k=top_k, dim=-1)
    top_k_logprobs = []
    # Position 0: empty placeholder
    top_k_logprobs.append([])
    # Positions 1 to seq_len-1: shifted from log_probs
    for i in range(seq_len - 1):
        position_topk = [
            (top_k_indices[i, j].item(), top_k_values[i, j].item())
            for j in range(top_k)
        ]
        top_k_logprobs.append(position_topk)

    # Compute exact entropy from logits before discarding
    # H(p) = -sum(p * log(p))
    probs = F.softmax(logits, dim=-1)
    unshifted_entropy = -(probs * log_probs).sum(dim=-1)
    # Shift to align with token positions: entropy[i] = H(logits[i-1])
    first_entropy = torch.tensor(
        [float("nan")], dtype=unshifted_entropy.dtype, device=unshifted_entropy.device
    )
    precomputed_entropy = torch.cat([first_entropy, unshifted_entropy[:-1]])

    return PrefillResult(
        seq_len=seq_len,
        token_logprobs=token_logprobs,
        top_k_logprobs=top_k_logprobs,
        input_ids=input_ids,
        precomputed_entropy=precomputed_entropy,
    )


def fireworks_to_prefill_result(
    tokens: List[str],
    token_ids: List[int],
    logprobs: List[Optional[float]],
    top_logprobs: List[List[Dict[str, Any]]],
) -> PrefillResult:
    """Convert Fireworks API response to unified PrefillResult.

    Args:
        tokens: List of token strings.
        token_ids: List of token IDs.
        logprobs: Log probability of each token.
        top_logprobs: Top-k logprobs as list of dicts with 'token_id', 'logprob'.

    Returns:
        PrefillResult (without full logits).
    """
    seq_len = len(token_ids)

    # Convert logprobs to tensor, using NaN for None
    token_logprobs_list = [lp if lp is not None else float("nan") for lp in logprobs]
    token_logprobs = torch.tensor(token_logprobs_list)

    # Convert top_logprobs to list of tuples
    top_k_logprobs = []
    for pos_topk in top_logprobs:
        position_topk = [
            (item.get("token_id", -1), item["logprob"])
            for item in pos_topk
        ]
        top_k_logprobs.append(position_topk)

    return PrefillResult(
        seq_len=seq_len,
        token_logprobs=token_logprobs,
        top_k_logprobs=top_k_logprobs,
        input_ids=torch.tensor(token_ids),
        full_logits=None,
    )


# =============================================================================
# Unified Metrics (work with PrefillResult)
# =============================================================================


class UnifiedMetric(ABC):
    """Base class for metrics that work with PrefillResult.

    Metrics can compute exact values when full_logits are available,
    or approximate values from top-k logprobs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name for reporting."""
        ...

    @property
    def is_shifted(self) -> bool:
        """Whether output is shifted (length = seq_len - 1)."""
        return False

    @property
    def requires_full_logits(self) -> bool:
        """Whether this metric needs full logits (no approximation possible)."""
        return False

    @abstractmethod
    def compute(self, result: PrefillResult) -> torch.Tensor:
        """Compute metric values.

        Args:
            result: Unified prefill result.

        Returns:
            Per-token metric values.
        """
        ...


class NegLogProbUnified(UnifiedMetric):
    """Negative log probability of each token.

    Works identically for both backends since we have exact token logprobs.
    """

    @property
    def name(self) -> str:
        return "neg_log_prob"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        return -result.token_logprobs


class PerplexityUnified(UnifiedMetric):
    """Per-token perplexity = exp(-log_prob).

    Works identically for both backends.
    """

    @property
    def name(self) -> str:
        return "perplexity"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        return torch.exp(-result.token_logprobs)


class EntropyUnified(UnifiedMetric):
    """Exact entropy of the probability distribution.

    Semantics: entropy[i] = H(distribution used to predict token_i)
    - Position 0: NaN (no prior context to compute entropy from)
    - Position i > 0: entropy of distribution at position i-1 (which predicts token_i)

    Uses precomputed_entropy for local models, returns NaN for API backends.
    """

    @property
    def name(self) -> str:
        return "entropy"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        if result.precomputed_entropy is not None:
            return result.precomputed_entropy
        return torch.full((result.seq_len,), float("nan"))


class TopKEntropyUnified(UnifiedMetric):
    """Entropy computed from renormalized top-k distribution.

    This metric works with both backends by using top_k_logprobs.
    The top-k probabilities are renormalized to sum to 1, then entropy is computed.

    Note: This is an approximation that ignores tail probability mass.
    It will be lower than true entropy, especially when top-k mass is small.

    Semantics: entropy[i] = H_topk(distribution used to predict token_i)
    """

    @property
    def name(self) -> str:
        return "topk_entropy"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        entropies = []
        for pos_topk in result.top_k_logprobs:
            if not pos_topk:
                entropies.append(float("nan"))
                continue
            logps = [lp for _, lp in pos_topk]
            # Renormalize using log-sum-exp for numerical stability
            max_logp = max(logps)
            probs = [math.exp(lp - max_logp) for lp in logps]
            z = sum(probs)
            probs = [p / z for p in probs]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropies.append(entropy)
        return torch.tensor(entropies)


class EstimatedEntropyUnified(UnifiedMetric):
    """Estimated entropy using tight lower bound from top-k.

    Uses the constraint that all tail tokens have probability <= p_k (the k-th
    highest probability) to compute a tighter lower bound on entropy.

    Given top-k probabilities p_1 >= p_2 >= ... >= p_k and tail mass r = 1 - sum(p_i):
    - To minimize tail entropy, pack mass into fewest tokens with prob <= p_k
    - n = ceil(r / p_k) tokens needed
    - (n-1) tokens have probability p_k, one has remainder

    H_tail_min = -(n-1) * p_k * log(p_k) - rem * log(rem)
    H >= H_k + H_tail_min

    With precomputed_entropy (local models): returns exact entropy.
    With API backends: returns tight lower bound from top-k.

    Semantics: entropy[i] = estimated H(distribution used to predict token_i)
    """

    @property
    def name(self) -> str:
        return "estimated_entropy"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        # Use precomputed entropy if available (local models)
        if result.precomputed_entropy is not None:
            return result.precomputed_entropy

        # Fall back to tight lower bound from top-k (API backends)
        entropies = []
        for pos_topk in result.top_k_logprobs:
            if not pos_topk:
                entropies.append(float("nan"))
                continue

            # Get probabilities from logprobs
            probs = [math.exp(lp) for _, lp in pos_topk]
            topk_mass = sum(probs)
            tail_mass = max(0.0, 1.0 - topk_mass)

            # Entropy from top-k (exact for these tokens)
            h_topk = -sum(p * math.log(p) for p in probs if p > 0)

            if tail_mass <= 1e-10:
                # No significant tail mass
                entropies.append(h_topk)
                continue

            # p_k is the smallest probability in top-k (last one, assuming sorted)
            p_k = min(probs)

            if p_k <= 1e-10:
                # p_k is too small, fall back to simple bound
                h_tail = -tail_mass * math.log(tail_mass) if tail_mass > 0 else 0.0
                entropies.append(h_topk + h_tail)
                continue

            # Tight lower bound: pack tail mass into fewest tokens with prob <= p_k
            # n = ceil(r / p_k)
            n = math.ceil(tail_mass / p_k)
            # rem = r - (n-1) * p_k
            rem = tail_mass - (n - 1) * p_k

            # H_tail_min = -(n-1) * p_k * log(p_k) - rem * log(rem)
            h_tail_min = 0.0
            if n > 1:
                h_tail_min -= (n - 1) * p_k * math.log(p_k)
            if rem > 1e-10:
                h_tail_min -= rem * math.log(rem)

            entropies.append(h_topk + h_tail_min)

        return torch.tensor(entropies)


class EntropyLowerBoundUnified(UnifiedMetric):
    """Simple lower bound on entropy using top-k mass + single tail bucket.

    This is a looser bound than EstimatedEntropyUnified.
    H >= H_k - r * log(r), where r is the tail mass.

    With precomputed_entropy (local models): returns exact entropy.
    With API backends: returns simple lower bound from top-k.

    Semantics: entropy[i] = lower bound on H(distribution used to predict token_i)
    """

    @property
    def name(self) -> str:
        return "entropy_lower"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        # Use precomputed entropy if available (local models)
        if result.precomputed_entropy is not None:
            return result.precomputed_entropy

        # Fall back to simple lower bound from top-k (API backends)
        entropies = []
        for pos_topk in result.top_k_logprobs:
            if not pos_topk:
                entropies.append(float("nan"))
                continue
            probs = [math.exp(lp) for _, lp in pos_topk]
            topk_mass = sum(probs)
            tail_mass = max(0.0, 1.0 - topk_mass)

            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            if tail_mass > 1e-10:
                entropy -= tail_mass * math.log(tail_mass)
            entropies.append(entropy)
        return torch.tensor(entropies)


class TopKMassUnified(UnifiedMetric):
    """Probability mass covered by top-k tokens.

    Useful for understanding approximation quality.
    """

    @property
    def name(self) -> str:
        return "topk_mass"

    def compute(self, result: PrefillResult) -> torch.Tensor:
        masses = []
        for pos_topk in result.top_k_logprobs:
            if not pos_topk:
                masses.append(float("nan"))
                continue
            mass = sum(math.exp(lp) for _, lp in pos_topk)
            masses.append(mass)
        return torch.tensor(masses)


class RankUnified(UnifiedMetric):
    """Rank of the actual token in the distribution.

    - With full_logits: exact rank
    - Without: rank within top-k (or k+1 if not in top-k)
    """

    @property
    def name(self) -> str:
        return "rank"

    @property
    def requires_full_logits(self) -> bool:
        return False  # Can approximate from top-k

    def compute(self, result: PrefillResult) -> torch.Tensor:
        if result.has_full_logits:
            return self._exact_rank(result.full_logits, result.input_ids)
        else:
            return self._approx_rank(result.top_k_logprobs, result.input_ids)

    def _exact_rank(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)
        ranks = torch.zeros(logits.size(0), dtype=torch.float)
        for i in range(logits.size(0)):
            rank = (sorted_indices[i] == input_ids[i]).nonzero(as_tuple=True)[0]
            ranks[i] = rank.item() + 1 if len(rank) > 0 else -1
        return ranks

    def _approx_rank(self, top_k_logprobs: List[List[tuple]], input_ids: torch.Tensor) -> torch.Tensor:
        ranks = []
        for i, pos_topk in enumerate(top_k_logprobs):
            token_id = input_ids[i].item()
            rank = len(pos_topk) + 1  # Default: not in top-k
            for j, (tid, _) in enumerate(pos_topk):
                if tid == token_id:
                    rank = j + 1
                    break
            ranks.append(rank)
        return torch.tensor(ranks, dtype=torch.float)


# Default unified metrics (works with both local and API backends)
DEFAULT_UNIFIED_METRICS = [
    NegLogProbUnified(),
    PerplexityUnified(),
    EstimatedEntropyUnified(),  # Uses tight lower bound for API, exact for local
]

# All entropy metrics for detailed analysis
ALL_ENTROPY_METRICS = [
    EntropyUnified(),           # Exact (requires full_logits)
    TopKEntropyUnified(),       # Renormalized top-k
    EstimatedEntropyUnified(),  # Tight lower bound
    EntropyLowerBoundUnified(), # Simple lower bound
]


def compute_unified_metrics(
    result: PrefillResult,
    metrics: Optional[List[UnifiedMetric]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute metrics from a unified PrefillResult.

    Args:
        result: Unified prefill result from either backend.
        metrics: List of metrics to compute. Defaults to neg_log_prob, perplexity, entropy.

    Returns:
        Dict mapping metric name to per-token values.
    """
    if metrics is None:
        metrics = DEFAULT_UNIFIED_METRICS

    return {m.name: m.compute(result) for m in metrics}


# =============================================================================
# Fireworks API Integration
# =============================================================================


def fireworks_prefill(
    token_ids: List[int],
    model: str = "accounts/fireworks/models/gpt-oss-20b",
    api_key: Optional[str] = None,
    top_logprobs: int = 5,
    timeout: int = 120,
) -> PrefillResult:
    """Prefill token IDs through Fireworks Completions API and get logprobs.

    Uses the Fireworks Completions API with echo=True to get logprobs
    for all input tokens. Accepts token IDs directly.

    Args:
        token_ids: List of token IDs to prefill.
        model: Fireworks model name (default: gpt-oss-20b).
        api_key: Fireworks API key. If None, reads from FIREWORKS_API_KEY env var.
        top_logprobs: Number of top logprobs per position (max 5 for Fireworks).
        timeout: Request timeout in seconds.

    Returns:
        PrefillResult with token-level statistics (no full_logits).

    Raises:
        RuntimeError: On API errors.
    """
    import requests

    if api_key is None:
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError(
                "FIREWORKS_API_KEY not set. Pass api_key or set environment variable."
            )

    url = "https://api.fireworks.ai/inference/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": token_ids,  # Pass token IDs directly
        "max_tokens": 1,  # Minimal generation (we only want prompt logprobs)
        "echo": True,  # Return prompt tokens with logprobs
        "logprobs": True,
        "top_logprobs": min(top_logprobs, 5),  # Fireworks max is 5
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    result = response.json()

    if "error" in result:
        raise RuntimeError(f"Fireworks API error: {result['error']}")

    content = result["choices"][0]["logprobs"]["content"]

    # Only keep prompt tokens (exclude the generated token at the end)
    prompt_len = len(token_ids)
    content = content[:prompt_len]

    tokens = []
    token_ids_out = []
    logprobs_list = []
    top_logprobs_list = []

    for token_info in content:
        tokens.append(token_info["token"])
        token_ids_out.append(token_info["token_id"])
        logprobs_list.append(token_info.get("logprob"))
        # Extract top logprobs as list of dicts
        top_lps = []
        if token_info.get("top_logprobs"):
            for tlp in token_info["top_logprobs"]:
                top_lps.append({
                    "token": tlp["token"],
                    "logprob": tlp["logprob"],
                    "token_id": tlp.get("token_id"),
                })
        top_logprobs_list.append(top_lps)

    # Convert to unified PrefillResult
    return fireworks_to_prefill_result(
        tokens=tokens,
        token_ids=token_ids_out,
        logprobs=logprobs_list,
        top_logprobs=top_logprobs_list,
    )


def fireworks_prefill_and_compute_metrics(
    token_ids: List[int],
    model: str = "accounts/fireworks/models/gpt-oss-20b",
    api_key: Optional[str] = None,
    top_logprobs: int = 5,
    timeout: int = 120,
    metrics: Optional[List[UnifiedMetric]] = None,
) -> Dict[str, torch.Tensor]:
    """Prefill token IDs through Fireworks and compute token-level metrics.

    Convenience function combining fireworks_prefill() and compute_unified_metrics().

    Args:
        token_ids: List of token IDs to prefill.
        model: Fireworks model name.
        api_key: Fireworks API key (or from env).
        top_logprobs: Number of top logprobs per position (max 5).
        timeout: Request timeout in seconds.
        metrics: List of UnifiedMetric objects to compute.

    Returns:
        Dict mapping metric name to per-token values tensor.
    """
    result = fireworks_prefill(
        token_ids=token_ids,
        model=model,
        api_key=api_key,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )
    return compute_unified_metrics(result, metrics)
