from __future__ import annotations

import textwrap
import matplotlib.pyplot as plt

# Consistent colors across all plots
COLOR_ACV = "green"
COLOR_HF = "blue"
COLOR_HF_BUDGET = "blue"
COLOR_HF_PAIRED = "purple"
COLOR_TRUE = "black"


# =============================================================================
# Shared plotting helpers for numerical experiments
# =============================================================================


def wrap_title(text: str, width: int = 62) -> str:
    """Split long plot titles across multiple lines for readability."""
    return "\n".join(textwrap.wrap(text, width=width))


def wrap_ylabel(text: str) -> str:
    """
    Insert line breaks in long y-axis labels so they do not crowd the title area.
    """
    replacements = {
        "Average estimated control-variate coefficient": "Average estimated\ncontrol-variate coefficient",
        "Average one-sided upper confidence bound": "Average one-sided\nupper confidence bound",
        "Average upper confidence bound": "Average upper\nconfidence bound",
        "Average one-sided margin of error": "Average one-sided\nmargin of error",
        "Average elapsed runtime (seconds)": "Average elapsed\nruntime (seconds)",
        "Empirical variance across macro-replications": "Empirical variance across\nmacro-replications",
        "Probability of realized improvement": "Probability of\nrealized improvement",
        "Average point estimate / average upper confidence bound": "Average point estimate /\naverage upper confidence bound",
    }
    return replacements.get(text, text)


def add_bottom_legend(fig, ax, yshift: float = -0.15):
    """
    Place a vertically stacked legend well below the x-axis label so figure
    width and title centering are unaffected.
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, yshift),
        ncol=1,
        frameon=True,
    )


def apply_grid(ax, with_minor: bool = True):
    """Apply consistent major/minor grid styling and show minor ticks."""
    ax.grid(True, which="major", alpha=0.35)
    if with_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.18, linestyle=":")
    ax.tick_params(axis="both", which="major", length=6)
    ax.tick_params(axis="both", which="minor", length=3)


def finalize_standard_plot(
    fig,
    ax,
    *,
    xlabel,
    ylabel,
    title,
    bottom_rect=(0.11, 0.46, 0.98, 0.95),
):
    """
    Standard final formatting for line plots.
    """
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(wrap_ylabel(ylabel), labelpad=14)
    ax.set_title(wrap_title(title), pad=14)
    add_bottom_legend(fig, ax, yshift=-0.15)
    # fig.tight_layout(rect=bottom_rect)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.42)


def compute_axis_limits_from_intervals(
    sub,
    *,
    acv_upper_col,
    hf_upper_col,
    hf_point_col,
    acv_point_col,
    true_gap,
    pad_fraction=0.08,
    default_pad=1.0,
):
    """Compute x-axis limits for interval comparison plots."""
    min_point = min(
        sub[hf_point_col].min(),
        sub[acv_point_col].min(),
        true_gap,
    )
    max_upper = max(
        sub[acv_upper_col].max(),
        sub[hf_upper_col].max(),
    )
    span = max_upper - min_point
    pad = pad_fraction * span if span > 0 else default_pad
    return min_point - pad, max_upper + pad


def draw_horizontal_interval(
    ax,
    *,
    y,
    ci_lower,
    ci_upper,
    point_estimate,
    x_left,
    line_color,
    line_style="-",
    line_width=2,
    marker="o",
    marker_color=None,
    marker_size=8,
    alpha=0.9,
    cap_half_height=0.08,
):
    """Draw one horizontal one-sided confidence interval and its point estimate."""
    visible_left = max(ci_lower, x_left)
    visible_right = ci_upper

    ax.hlines(
        y=y,
        xmin=visible_left,
        xmax=visible_right,
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
        alpha=alpha,
    )

    if ci_lower >= x_left:
        ax.vlines(
            x=ci_lower,
            ymin=y - cap_half_height,
            ymax=y + cap_half_height,
            color=line_color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=alpha,
        )

    ax.vlines(
        x=visible_right,
        ymin=y - cap_half_height,
        ymax=y + cap_half_height,
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
        alpha=alpha,
    )

    ax.plot(
        point_estimate,
        y,
        marker=marker,
        color=marker_color if marker_color is not None else line_color,
        markersize=marker_size,
    )


def finalize_interval_plot(
    fig,
    ax,
    *,
    true_gap,
    x_left,
    x_right,
    y_positions,
    ytick_labels,
    xlabel,
    ylabel,
    title,
    legend_handles,
):
    """
    Apply common formatting to horizontal interval plots.
    """
    ax.axvline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(wrap_ylabel(ylabel), labelpad=14)
    ax.set_title(wrap_title(title), pad=14)
    apply_grid(ax, with_minor=True)

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=True,
    )
    # fig.tight_layout(rect=[0.10, 0.44, 0.98, 0.96])
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.40)


def make_standard_figure():
    """
    Create a taller standard line-plot figure so the y-axis has more room and
    the lower legend does not crowd the axes.
    """
    return plt.subplots(figsize=(9.5, 10.5))


def make_interval_figure(nrows_like: int):
    """
    Create a tall interval-comparison figure so the y-axis entries are not
    vertically compressed.
    """
    height = max(10.5, 8.8 + 1.05 * nrows_like)
    return plt.subplots(figsize=(12.0, height))
