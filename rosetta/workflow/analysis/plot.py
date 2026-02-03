"""Visualization utilities for perplexity analysis results."""

import csv
import gzip
import math
import random
from pathlib import Path
from typing import Dict, IO, List, Optional, Tuple
from zlib import crc32

# Type hints for the dataclasses from perplexity.py
# We use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from script.workflow.analysis.perplexity import (
        AggregatedMetrics,
        AnalysisResult,
        ComparisonResult,
    )


def plot_metric_by_role(
    aggregated: "AggregatedMetrics",
    metric_name: str,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
):
    """Bar plot of metric by message role.

    Args:
        aggregated: Aggregated metrics.
        metric_name: Name of metric to plot.
        output_path: Path to save the plot.
        title: Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    roles = []
    values = []
    for role, metrics in aggregated.by_role.items():
        if metric_name in metrics:
            roles.append(role)
            values.append(metrics[metric_name])

    if not roles:
        print(f"No data for metric {metric_name}")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(roles, values, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    plt.xlabel("Message Role")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title or f"{metric_name.replace('_', ' ').title()} by Message Role")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_metric_distribution(
    results: List["AnalysisResult"],
    metric_name: str,
    output_path: Optional[Path] = None,
    by_role: bool = True,
):
    """Plot distribution of metric values.

    Args:
        results: List of analysis results.
        metric_name: Name of metric to plot.
        output_path: Path to save the plot.
        by_role: Whether to separate by role.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    if by_role:
        role_values: Dict[str, List[float]] = {}
        for result in results:
            for section in result.sections:
                if metric_name in section.metrics:
                    if section.role not in role_values:
                        role_values[section.role] = []
                    role_values[section.role].append(section.metrics[metric_name])

        plt.figure(figsize=(12, 6))
        data = [values for values in role_values.values()]
        labels = list(role_values.keys())
        plt.boxplot(data, labels=labels)
        plt.xlabel("Message Role")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"{metric_name.replace('_', ' ').title()} Distribution by Role")
    else:
        all_values = []
        for result in results:
            if metric_name in result.overall_metrics:
                all_values.append(result.overall_metrics[metric_name])

        plt.figure(figsize=(10, 6))
        plt.hist(all_values, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel(metric_name.replace("_", " ").title())
        plt.ylabel("Count")
        plt.title(f"Distribution of {metric_name.replace('_', ' ').title()}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    comparisons: List["ComparisonResult"],
    metric_name: str,
    output_path: Optional[Path] = None,
):
    """Plot before/after comparison.

    Args:
        comparisons: List of comparison results.
        metric_name: Name of metric to plot.
        output_path: Path to save the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available for plotting")
        return

    original = [c.original_metrics.get(metric_name, 0) for c in comparisons]
    transformed = [c.transformed_metrics.get(metric_name, 0) for c in comparisons]

    x = np.arange(len(comparisons))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, original, width, label="Original", color="#3498db")
    plt.bar(x + width / 2, transformed, width, label="Transformed", color="#e74c3c")

    plt.xlabel("Conversation")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"{metric_name.replace('_', ' ').title()}: Original vs Transformed")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_metric_by_position(
    results: List["AnalysisResult"],
    metric_name: str,
    output_path: Optional[Path] = None,
    bin_size: int = 50,
    show_range: bool = True,
    title: Optional[str] = None,
    is_shifted: Optional[bool] = None,
):
    """Plot metric values by token position, colored by role.

    X-axis: Token position (binned)
    Y-axis: Mean metric value with range (shaded area for std)
    Legend: Different colors for different roles (system, user, assistant, tool)

    Args:
        results: List of analysis results with per-token metrics.
        metric_name: Name of metric to plot (e.g., "entropy", "perplexity").
        output_path: Path to save the plot. If None, displays interactively.
        bin_size: Number of tokens per bin for smoothing. Default 50.
        show_range: If True, show shaded region for std deviation.
        title: Custom plot title.
        is_shifted: If True, metric predicts next token (length seq_len-1).
            If None, auto-detects from metric length vs token_count.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available for plotting")
        return

    # Role colors
    role_colors = {
        "system": "#2ecc71",     # Green
        "user": "#3498db",       # Blue
        "assistant": "#9b59b6",  # Purple
        "tool": "#e74c3c",       # Red
    }

    # Collect per-position values grouped by role
    # Structure: {role: {position: [values]}}
    role_position_values: Dict[str, Dict[int, List[float]]] = {}

    for result in results:
        if metric_name not in result.metrics_by_position:
            continue

        metric_values = result.metrics_by_position[metric_name]

        # Build position -> role mapping from sections
        position_to_role = {}
        for section in result.sections:
            for pos in range(section.start_idx, section.end_idx):
                position_to_role[pos] = section.role

        # Some metrics (perplexity, neg_log_prob) are shifted by 1
        # They have length seq_len-1 and predict position i+1 from position i
        # So metric_values[i] corresponds to predicting token at position i+1
        detected_shifted = (
            is_shifted if is_shifted is not None else len(metric_values) < result.token_count
        )

        for i, value in enumerate(metric_values):
            # For shifted metrics, the value at index i predicts token i+1
            # So we assign it to position i+1's role
            pos = i + 1 if detected_shifted else i

            if pos not in position_to_role:
                continue

            role = position_to_role[pos]

            if role not in role_position_values:
                role_position_values[role] = {}
            if pos not in role_position_values[role]:
                role_position_values[role][pos] = []

            # Handle inf/nan values
            if np.isfinite(value):
                role_position_values[role][pos].append(value)

    if not role_position_values:
        print(f"No data for metric {metric_name}")
        return

    # Bin the positions and compute statistics
    plt.figure(figsize=(14, 7))

    max_pos = max(
        max(positions.keys()) if positions else 0
        for positions in role_position_values.values()
    )

    for role in ["system", "user", "assistant", "tool"]:
        if role not in role_position_values:
            continue

        positions = role_position_values[role]
        if not positions:
            continue

        # Create bins
        num_bins = (max_pos // bin_size) + 1
        bin_means = []
        bin_stds = []
        bin_centers = []

        for bin_idx in range(num_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size

            # Collect all values in this bin for this role
            bin_values = []
            for pos in range(bin_start, bin_end):
                if pos in positions:
                    bin_values.extend(positions[pos])

            if bin_values:
                bin_centers.append((bin_start + bin_end) / 2)
                bin_means.append(np.mean(bin_values))
                bin_stds.append(np.std(bin_values))

        if not bin_centers:
            continue

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        color = role_colors.get(role, "#95a5a6")

        # Plot mean line
        plt.plot(bin_centers, bin_means, color=color, label=role, linewidth=2)

        # Plot shaded region for std
        if show_range and len(bin_centers) > 1:
            plt.fill_between(
                bin_centers,
                bin_means - bin_stds,
                bin_means + bin_stds,
                color=color,
                alpha=0.2,
            )

    plt.xlabel("Token Position")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title or f"{metric_name.replace('_', ' ').title()} by Token Position")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_metric_by_position_overlay(
    results: List["AnalysisResult"],
    metric_name: str,
    output_path: Optional[Path] = None,
    max_conversations: int = 10,
    title: Optional[str] = None,
    is_shifted: Optional[bool] = None,
):
    """Plot individual conversation traces overlaid, colored by role segments.

    Shows each conversation as a separate line with color-coded segments
    based on message role. Useful for seeing individual patterns.

    Args:
        results: List of analysis results.
        metric_name: Name of metric to plot.
        output_path: Path to save the plot.
        max_conversations: Maximum number of conversations to overlay.
        title: Custom plot title.
        is_shifted: If True, metric predicts next token (length seq_len-1).
            If None, auto-detects from metric length vs token_count.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available for plotting")
        return

    role_colors = {
        "system": "#2ecc71",
        "user": "#3498db",
        "assistant": "#9b59b6",
        "tool": "#e74c3c",
    }

    plt.figure(figsize=(14, 7))

    # Track which roles we've added to legend
    legend_added = set()

    for idx, result in enumerate(results[:max_conversations]):
        if metric_name not in result.metrics_by_position:
            continue

        metric_values = result.metrics_by_position[metric_name]
        detected_shifted = (
            is_shifted if is_shifted is not None else len(metric_values) < result.token_count
        )

        # Plot each section with its role color
        for section in result.sections:
            start = section.start_idx
            end = section.end_idx

            # Adjust for shifted metrics
            if detected_shifted:
                start = max(0, start - 1)
                end = min(len(metric_values), end - 1)

            if end <= start:
                continue

            positions = np.arange(start, end)
            values = metric_values[start:end]

            # Filter out inf/nan
            mask = np.isfinite(values)
            if not np.any(mask):
                continue

            color = role_colors.get(section.role, "#95a5a6")
            label = section.role if section.role not in legend_added else None

            plt.plot(
                positions[mask],
                np.array(values)[mask],
                color=color,
                alpha=0.5,
                linewidth=0.8,
                label=label,
            )

            if label:
                legend_added.add(section.role)

    plt.xlabel("Token Position")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title or f"{metric_name.replace('_', ' ').title()} by Position (Individual Traces)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def _open_csv_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _reservoir_add(
    rng: random.Random,
    xs: List[int],
    ys: List[float],
    seen: int,
    x: int,
    y: float,
    k: int,
) -> int:
    """Reservoir sampling: keep a uniform sample of size k from a stream.

    Returns updated seen count.
    """
    seen += 1
    if k <= 0:
        return seen
    if len(xs) < k:
        xs.append(x)
        ys.append(y)
        return seen
    j = rng.randrange(seen)
    if j < k:
        xs[j] = x
        ys[j] = y
    return seen


def plot_metric_scatter_subplots_by_role_from_csv(
    csv_path: Path,
    metric_name: str,
    output_path: Optional[Path] = None,
    sample_per_role: int = 200_000,
    include_unknown: bool = True,
    title: Optional[str] = None,
    seed: int = 0,
    ncols: int = 2,
    roles: Optional[List[str]] = None,
    line_bin_size: int = 50,
    running_window_bins: int = 20,
):
    """Scatter plots in subplots, one per role.

    Reads the token-level CSV produced by `save_token_plot_data_csv()`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    role_colors = {
        "system": "#2ecc71",     # Green
        "user": "#3498db",       # Blue
        "assistant": "#9b59b6",  # Purple
        "tool": "#e74c3c",       # Red
        "unknown": "#95a5a6",    # Gray
    }

    # role -> (xs, ys, seen, rng) for scatter sampling
    series: Dict[str, Tuple[List[int], List[float], int, random.Random]] = {}
    # role -> (sum, count) for global mean line
    role_totals: Dict[str, Tuple[float, int]] = {}
    # role -> bin_idx -> (sum, count) for running average line
    role_bins: Dict[str, Dict[int, Tuple[float, int]]] = {}

    with _open_csv_text(csv_path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or metric_name not in reader.fieldnames:
            raise ValueError(
                f"CSV {csv_path} does not contain metric column {metric_name!r}. "
                f"Columns: {reader.fieldnames}"
            )

        for row in reader:
            role = (row.get("role") or "unknown").strip() or "unknown"
            if role == "unknown" and not include_unknown:
                continue
            if roles is not None and role not in set(roles):
                continue

            v_raw = row.get(metric_name, "")
            if not v_raw:
                continue
            try:
                v = float(v_raw)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue

            try:
                token_idx = int(row.get("token_idx") or 0)
            except ValueError:
                continue

            # Global mean accumulation
            total_sum, total_count = role_totals.get(role, (0.0, 0))
            role_totals[role] = (total_sum + v, total_count + 1)

            # Bin accumulation for running average line
            bin_idx = token_idx // max(1, int(line_bin_size))
            bins = role_bins.get(role)
            if bins is None:
                bins = {}
                role_bins[role] = bins
            bsum, bcount = bins.get(bin_idx, (0.0, 0))
            bins[bin_idx] = (bsum + v, bcount + 1)

            if role not in series:
                xs: List[int] = []
                ys: List[float] = []
                role_seed = seed ^ int(crc32(role.encode("utf-8")))
                series[role] = (xs, ys, 0, random.Random(role_seed))

            xs, ys, seen, rng = series[role]
            seen = _reservoir_add(rng, xs, ys, seen, token_idx, v, sample_per_role)
            series[role] = (xs, ys, seen, rng)

    if not series:
        print(f"No data found for metric {metric_name} in {csv_path}")
        return

    ordered_roles = ["system", "user", "assistant", "tool"]
    if include_unknown:
        ordered_roles.append("unknown")
    plot_roles = [r for r in ordered_roles if r in series and series[r][0]]
    plot_roles += sorted(set(series.keys()) - set(plot_roles))
    if not plot_roles:
        print(f"No non-empty role series for metric {metric_name} in {csv_path}")
        return

    ncols = max(1, int(ncols))
    nrows = (len(plot_roles) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, max(4, 3.8 * nrows)),
        sharex=True,
        sharey=True,
    )
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]

    window_bins = max(1, int(running_window_bins))

    for ax, role in zip(axes_list, plot_roles, strict=False):
        xs, ys, _, _ = series[role]
        ax.scatter(
            xs,
            ys,
            s=2,
            alpha=0.15,
            c=role_colors.get(role, "#95a5a6"),
            linewidths=0,
        )

        # Global mean (horizontal)
        total_sum, total_count = role_totals.get(role, (0.0, 0))
        if total_count > 0:
            mean_val = total_sum / total_count
            ax.axhline(
                mean_val,
                color=role_colors.get(role, "#95a5a6"),
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="mean",
            )

        # Running average line over binned means
        bins = role_bins.get(role) or {}
        if bins:
            sorted_bins = sorted(bins.items())
            x_centers: List[float] = []
            bin_means: List[float] = []
            for bidx, (bsum, bcount) in sorted_bins:
                if bcount <= 0:
                    continue
                x_centers.append((bidx + 0.5) * max(1, int(line_bin_size)))
                bin_means.append(bsum / bcount)

            if len(x_centers) >= 2:
                running: List[float] = []
                for i in range(len(bin_means)):
                    lo = max(0, i - window_bins + 1)
                    running.append(sum(bin_means[lo : i + 1]) / (i + 1 - lo))
                ax.plot(
                    x_centers,
                    running,
                    color="black",
                    linewidth=2.0,
                    alpha=0.85,
                    label=f"running avg ({window_bins} bins)",
                )

        ax.set_title(f"{role} (n={len(xs)})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=True)

    for ax in axes_list[len(plot_roles):]:
        ax.axis("off")

    fig.supxlabel("Token Position (within conversation)")
    fig.supylabel(metric_name.replace("_", " ").title())
    fig.suptitle(title or f"{metric_name.replace('_', ' ').title()} Scatter by Role", y=1.02)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)
