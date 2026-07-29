"""Figures for the statistical report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .metrics import roc_auc, roc_curve  # noqa: E402

__all__ = [
    "PUBLICATION_STYLE",
    "plot_auc_distribution",
    "plot_calibration",
    "plot_confusion_matrices",
    "plot_mean_roc",
]

TOP1 = "Validation-selected Top-1 Individual"
TOP3 = "Validation-selected Top-3 Average Ensemble"
ALL6 = "All-6 Average Ensemble"

# Fixed, colour-blind-safe model identity across every figure.
MODEL_COLOURS = {
    "ResNet50": "#7C83A3",
    "DenseNet121": "#66A182",
    "InceptionV3": "#C98B5B",
    "ConvNeXtV2": "#8E6C9F",
    "ViT": "#C95D6A",
    "SwinT": "#5D9C99",
    TOP1: "#747474",
    TOP3: "#2F557F",
    ALL6: "#C49A32",
}

CONFUSION_COLOURS = LinearSegmentedColormap.from_list(
    "publication_navy",
    ("#F4F6F8", "#C8D5E3", MODEL_COLOURS[TOP3]),
)

DISPLAY_NAMES = {
    TOP1: "Validation-selected Top-1",
    TOP3: "Validation-selected Top-3 ensemble",
    ALL6: "All-6 ensemble",
}

LINE_STYLES = {
    TOP1: (0, (5, 2)),
    TOP3: "-",
    ALL6: (0, (1.5, 1.5)),
}

PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "text.color": "black",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.labelcolor": "black",
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_GRID = np.linspace(0.0, 1.0, 10_001)


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _colour(name: str) -> str:
    return MODEL_COLOURS.get(name, "#333333")


def _style_axes(axis: plt.Axes, grid_axis: str | None = None) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    axis.tick_params(direction="out", length=3)
    if grid_axis is not None:
        axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)


def _save(figure: plt.Figure, destination: Path) -> None:
    figure.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def _interpolate_roc(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Interpolate one empirical ROC while preserving its vertical segments."""
    _, sensitivity, specificity = roc_curve(y_true, scores)
    fpr = (1.0 - specificity)[::-1]
    tpr = sensitivity[::-1]

    unique_fpr, starts, counts = np.unique(fpr, return_index=True, return_counts=True)
    lower_tpr = tpr[starts]
    upper_tpr = tpr[starts + counts - 1]

    interval = np.searchsorted(unique_fpr, _GRID, side="right") - 1
    interval = np.clip(interval, 0, len(unique_fpr) - 2)
    left_fpr = unique_fpr[interval]
    right_fpr = unique_fpr[interval + 1]
    weight = (_GRID - left_fpr) / (right_fpr - left_fpr)
    interpolated = upper_tpr[interval] + weight * (lower_tpr[interval + 1] - upper_tpr[interval])
    interpolated[unique_fpr[-1] <= _GRID] = upper_tpr[-1]
    return np.clip(interpolated, 0.0, 1.0)


def _mean_roc(y_true: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Mean of repetition-level ROC curves on a common false-positive-rate grid."""
    curves = []
    for column in range(matrix.shape[1]):
        curves.append(_interpolate_roc(y_true, matrix[:, column]))
    return np.vstack(curves).mean(axis=0)


def plot_mean_roc(y_true: np.ndarray, matrices: dict[str, np.ndarray], destination: Path) -> None:
    """Mean repetition-level ROC curves without inferential bands."""
    with plt.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(figsize=(7.4, 5.8))
        legend_entries = []
        for name, matrix in matrices.items():
            mean = _mean_roc(y_true, matrix)
            mean_auc = np.mean(
                [roc_auc(y_true, matrix[:, column]) for column in range(matrix.shape[1])]
            )
            is_selected = name in {TOP1, TOP3}
            (line,) = axes.plot(
                _GRID,
                mean,
                color=_colour(name),
                linestyle=LINE_STYLES.get(name, "-"),
                linewidth=2.4 if name == TOP3 else 2.0 if is_selected else 1.35,
                alpha=1.0 if name in {TOP1, TOP3, ALL6} else 0.9,
                label=f"{_display_name(name)} (AUC = {mean_auc:.3f})",
                zorder=4 if is_selected else 3,
            )
            legend_entries.append((mean_auc, line))

        axes.plot(
            [0, 1],
            [0, 1],
            linestyle=(0, (4, 3)),
            color="#8C8C8C",
            linewidth=1.0,
            zorder=2,
        )
        axes.set(
            xlabel="False-positive rate",
            xlim=(0, 1),
            ylabel="True-positive rate",
            ylim=(0, 1),
        )
        axes.set_aspect("equal")
        axes.set_xticks(np.linspace(0, 1, 6))
        axes.set_yticks(np.linspace(0, 1, 6))
        _style_axes(axes, "both")
        legend_entries.sort(key=lambda entry: entry[0], reverse=True)
        handles = [entry[1] for entry in legend_entries]
        axes.legend(
            handles=handles,
            loc="lower right",
            handlelength=2.6,
            frameon=False,
        )
        _save(figure, destination)


def plot_auc_distribution(results: pd.DataFrame, destination: Path) -> None:
    """Box plot of per-repetition AUC, ordered by mean."""
    order = results.groupby("Model")["AUC"].mean().sort_values(ascending=False).index
    data = [results.loc[results["Model"] == name, "AUC"].to_numpy() for name in order]

    with plt.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(figsize=(7.2, 5.8))
        boxes = axes.boxplot(
            data,
            tick_labels=[_display_name(name) for name in order],
            orientation="horizontal",
            showfliers=False,
            patch_artist=True,
            widths=0.58,
            medianprops={"color": "white", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )
        rng = np.random.default_rng(0)
        for position, (name, values, box) in enumerate(
            zip(order, data, boxes["boxes"], strict=True), start=1
        ):
            colour = _colour(name)
            box.set(facecolor=colour, edgecolor=colour, alpha=0.85, linewidth=0.9)
            jitter = rng.normal(0, 0.045, size=len(values))
            axes.scatter(
                values,
                position + jitter,
                s=13,
                facecolor="white",
                edgecolor=colour,
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
            )
        lower = max(0.0, float(results["AUC"].min()) - 0.02)
        axes.set(xlabel="Area under the ROC curve", xlim=(lower, 1.0))
        axes.invert_yaxis()
        _style_axes(axes, "x")
        _save(figure, destination)


def plot_calibration(tables: dict[str, pd.DataFrame], destination: Path) -> None:
    """Reliability diagram: observed frequency against predicted probability."""
    markers = {"ViT": "s", TOP3: "o", ALL6: "^"}
    with plt.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(figsize=(5.8, 5.4))
        axes.plot(
            [0, 1],
            [0, 1],
            linestyle=(0, (4, 3)),
            color="#8C8C8C",
            linewidth=1.0,
            label="Perfect calibration",
        )
        for name, table in tables.items():
            axes.plot(
                table["mean_predicted"],
                table["observed_frequency"],
                marker=markers.get(name, "o"),
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linewidth=2.0 if name == TOP3 else 1.7,
                linestyle=LINE_STYLES.get(name, "-"),
                color=_colour(name),
                label=_display_name(name),
            )
        axes.set(
            xlabel="Mean predicted probability",
            xlim=(0, 1),
            ylabel="Observed thrombus frequency",
            ylim=(0, 1),
        )
        axes.set_aspect("equal")
        axes.set_xticks(np.linspace(0, 1, 6))
        axes.set_yticks(np.linspace(0, 1, 6))
        _style_axes(axes, "both")
        axes.legend(loc="upper left", handlelength=2.6)
        _save(figure, destination)


def plot_confusion_matrices(tables: dict[str, pd.DataFrame], destination: Path) -> None:
    """Row-normalised confusion matrices averaged across repetitions."""
    with plt.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(
            1,
            len(tables),
            figsize=(4.2 * len(tables), 3.7),
            squeeze=False,
            constrained_layout=True,
        )
        figure.suptitle("Mean confusion matrices across 10 repetitions", fontsize=10)
        for panel, (axis, (name, table)) in enumerate(zip(axes[0], tables.items(), strict=True)):
            counts = table[["TN", "FP", "FN", "TP"]].mean().to_numpy().reshape(2, 2)
            denominators = counts.sum(axis=1, keepdims=True)
            normalised = np.divide(
                counts, denominators, out=np.zeros_like(counts), where=denominators != 0
            )
            axis.imshow(normalised, cmap=CONFUSION_COLOURS, vmin=0, vmax=1)
            for row in range(2):
                for column in range(2):
                    axis.text(
                        column,
                        row,
                        f"{normalised[row, column]:.1%}\nMean count = {counts[row, column]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        linespacing=1.35,
                        color="white" if normalised[row, column] > 0.55 else "#1A1A1A",
                    )
            axis.set(
                title=_display_name(name),
                xlabel="Predicted class",
                ylabel="True class" if panel == 0 else "",
                xticks=(0, 1),
                yticks=(0, 1),
                xticklabels=("Negative", "Positive"),
                yticklabels=("Negative", "Positive"),
            )
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)
        _save(figure, destination)
