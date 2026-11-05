"""README charts, drawn in a light and a dark version from the saved run results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33332f",
        "series": ["#3987e5", "#d95926"],
    },
}


def styled_axes(theme: dict, ncols: int = 1, width: float = 7.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 3.4), squeeze=False, sharey=True)
    fig.patch.set_facecolor(theme["surface"])
    for ax in axes[0]:
        ax.set_facecolor(theme["surface"])
        ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
        ax.tick_params(colors=theme["muted"], length=0, labelsize=8)
        ax.xaxis.label.set_color(theme["muted"])
        ax.yaxis.label.set_color(theme["muted"])
    return fig, axes[0]


def title(ax, theme: dict, text: str) -> None:
    ax.set_title(text, loc="left", color=theme["text"], fontsize=10, fontweight="bold", pad=10)


def percent_axis(ax) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def save(fig, out_dir: Path, name: str, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}-{mode}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path}")


def plot_sweep(sweep: dict, out_dir: Path, mode: str) -> None:
    theme = THEMES[mode]
    panels = [
        ("ambiguous_destructive", "Destructive calls on ambiguous tasks (lower is better)"),
        ("explicit_delete", "Clear delete requests done (higher is better)"),
    ]
    fig, axes = styled_axes(theme, ncols=2, width=11)
    alphas = sorted({row["alpha"] for row in sweep["rows"]}, reverse=True)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        title(ax, theme, label)
        ax.axhline(
            sweep["baseline"][metric],
            color=theme["muted"],
            linewidth=1.2,
            linestyle="--",
            label="base model",
        )
        for color, alpha in zip(theme["series"], alphas, strict=False):
            rows = [row for row in sweep["rows"] if row["alpha"] == alpha]
            layers = [row["layer"] for row in rows]
            values = [row[metric] for row in rows]
            ax.plot(
                layers,
                values,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                markeredgecolor=theme["surface"],
                markeredgewidth=1,
                label=f"alpha = {alpha:g}",
            )
        best = sweep["best"]
        ax.plot(
            best["layer"],
            best[metric],
            marker="o",
            markersize=9,
            fillstyle="none",
            color=theme["text"],
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"picked: layer {best['layer']}",
            (best["layer"], best[metric]),
            xytext=(12, -3),
            textcoords="offset points",
            ha="left",
            va="center",
            color=theme["text"],
            fontsize=8,
        )
        ax.set_xlabel("Layer the direction comes from")
        ax.set_xticks(range(4, 25, 2))
        percent_axis(ax)
    legend = axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncols=3)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    fig.tight_layout(w_pad=3)
    save(fig, out_dir, "sweep", mode)


def plot_refusal_cosine(analysis: dict, out_dir: Path, mode: str, hidden_size: int = 1536) -> None:
    theme = THEMES[mode]
    layers = analysis["refusal_cosine"]["layers"]
    cosines = dict(zip(layers, analysis["refusal_cosine"]["cosine"], strict=True))
    fig, (ax,) = styled_axes(theme)
    title(ax, theme, "Cosine similarity between the destructive and refusal directions")
    # Two random unit vectors in d dimensions have cosine ~ N(0, 1/d). Shade +-2 SD.
    chance = 2 / hidden_size**0.5
    ax.axhspan(-chance, chance, color=theme["grid"], linewidth=0)
    ax.text(
        layers[-1],
        -chance,
        "random pairs, 95% range",
        ha="right",
        va="top",
        color=theme["muted"],
        fontsize=8,
    )
    ax.axhline(0, color=theme["muted"], linewidth=1)
    ax.plot(
        layers,
        list(cosines.values()),
        color=theme["series"][0],
        linewidth=2,
        marker="o",
        markersize=4,
        markeredgecolor=theme["surface"],
        markeredgewidth=1,
    )
    picked = analysis["layer"]
    ax.plot(
        picked,
        cosines[picked],
        marker="o",
        markersize=9,
        fillstyle="none",
        color=theme["text"],
        markeredgewidth=1.5,
    )
    ax.annotate(
        f"layer {picked}: {cosines[picked]:.2f}",
        (picked, cosines[picked]),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        color=theme["text"],
        fontsize=8,
    )
    ax.set_ylim(-0.25, 0.5)
    ax.set_yticks([-0.25, 0, 0.25, 0.5])
    ax.set_xticks(range(1, layers[-1] + 1, 3))
    ax.set_xlabel("Layer")
    fig.tight_layout()
    save(fig, out_dir, "refusal-cosine", mode)


def plot_controls(analysis: dict, out_dir: Path, mode: str) -> None:
    theme = THEMES[mode]
    intervals = analysis["intervals"]
    rows = [
        (
            "Base model",
            [intervals["base"]["ambiguous_destructive"]["rate"]],
            intervals["base"]["ambiguous_destructive"]["ci95"],
        ),
        (
            f"Learned direction (layer {analysis['layer']})",
            [intervals["edited"]["ambiguous_destructive"]["rate"]],
            intervals["edited"]["ambiguous_destructive"]["ci95"],
        ),
        (
            "Random directions (5)",
            [r["ambiguous_destructive"] for r in analysis["controls"]["random"]],
            None,
        ),
        (
            "Shuffled-label directions (5)",
            [r["ambiguous_destructive"] for r in analysis["controls"]["shuffled"]],
            None,
        ),
    ]
    fig, (ax,) = styled_axes(theme)
    title(ax, theme, "Destructive calls on ambiguous tasks, by what was removed")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    for y, (_, values, ci) in enumerate(reversed(rows)):
        if ci is not None:
            ax.plot(ci, [y, y], color=theme["series"][0], linewidth=2, solid_capstyle="round")
        # Stack repeated values upward so every run stays visible.
        offsets = [0.12 * values[:i].count(v) for i, v in enumerate(values)]
        ax.scatter(
            values,
            [y + o for o in offsets],
            s=48,
            color=theme["series"][0],
            edgecolors=theme["surface"],
            linewidths=1.5,
            zorder=3,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in reversed(rows)], color=theme["text"], fontsize=9)
    ax.set_xlim(-0.02, 0.62)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticklabels([f"{x:.0%}" for x in ax.get_xticks()])
    ax.set_xlabel("Share of 40 ambiguous tasks with a destructive command (bars: 95% interval)")
    fig.tight_layout()
    save(fig, out_dir, "controls", mode)


def plot_all(sweep: dict, analysis: dict, out_dir: Path) -> None:
    for mode in THEMES:
        plot_sweep(sweep, out_dir, mode)
        plot_refusal_cosine(analysis, out_dir, mode)
        plot_controls(analysis, out_dir, mode)
