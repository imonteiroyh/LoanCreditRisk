"""Utilities for segment tables, lift curves, gain curves, and KS statistics.

Adapted from https://github.com/tensorbored/kds with simplified naming."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLUMN_LABELS = {
    "probability_min": "Minimum probability in segment",
    "probability_max": "Maximum probability in segment",
    "probability_average": "Average probability in segment",
    "count": "Observations in segment",
    "responders": "Positive class count",
    "non_responders": "Negative class count",
    "expected_responders_random": "Expected responders if assignments were random",
    "expected_responders_perfect": "Expected responders under perfect ranking",
    "response_rate": "Response rate (%) in segment",
    "cumulative_count": "Cumulative observations",
    "cumulative_responders": "Cumulative responders",
    "cumulative_perfect_responders": "Cumulative perfect responders",
    "cumulative_non_responders": "Cumulative non-responders",
    "cumulative_count_percentage": "Cumulative observation percentage",
    "cumulative_responders_topdown": "Cumulative events when approving (lowest to highest risk)",
    "cumulative_count_topdown": "Cumulative observations when approving (lowest to highest risk)",
    "cumulative_count_percentage_topdown": "Cumulative observation percentage when approving (lowest to highest risk)",
    "cumulative_perfect_responders_topdown": "Cumulative events under perfect ranking (lowest to highest risk)",
    "cumulative_responders_rate": "Cumulative event rate of approved population (lowest to highest risk)",
    "cumulative_perfect_responders_rate": "Cumulative event rate under perfect ranking (lowest to highest risk)",
    "cumulative_random_responders_rate": "Baseline event rate under random selection",
    "cumulative_responders_percentage": "Cumulative responder percentage",
    "cumulative_perfect_responders_percentage": "Cumulative perfect responder percentage",
    "cumulative_non_responders_percentage": "Cumulative non-responder percentage",
    "ks_statistic": "KS statistic per segment",
    "lift": "Cumulative lift",
    "cumulative_count_percentage_topdown": "Cumulative observation percentage from the top segment downward",
    "cumulative_responders_rate": "Cumulative default rate (%) from the top segment downward",
    "cumulative_random_responders_rate": "Random selection default rate (%)",
    "cumulative_perfect_responders_rate": "Perfect model default rate (%)",
    "segment_default_rate": "Default rate (%) in segment",
    "segment_default_rate_random": "Random baseline default rate (%)",
}


def _normalize_y_prob(y_prob, names=None):
    """
    Normalize y_prob to dict[str, np.ndarray].
    """
    if isinstance(y_prob, dict):
        return {k: np.asarray(v) for k, v in y_prob.items()}

    if isinstance(y_prob, pd.DataFrame):
        return {col: y_prob[col].to_numpy() for col in y_prob.columns}

    if isinstance(y_prob, (list, tuple)):
        if names is None:
            names = [f"Model {i + 1}" for i in range(len(y_prob))]
        if len(names) != len(y_prob):
            raise ValueError("names must have same length as y_prob list/tuple.")
        return {n: np.asarray(v) for n, v in zip(names, y_prob)}

    return {"Model": np.asarray(y_prob)}


def print_labels():
    """Print the segment table column descriptions."""
    for key, description in COLUMN_LABELS.items():
        print(f"{key:<40} {description}")


def _segment_table_single(y_true, y_prob, n_segments=20, round_decimal=3):
    """Build the segment table for a single set of probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["segment"] = np.linspace(1, n_segments + 1, len(df), endpoint=False, dtype=int)

    grouped = df.groupby("segment")
    st = grouped.agg(
        probability_min=("y_prob", "min"),
        probability_max=("y_prob", "max"),
        probability_average=("y_prob", "mean"),
        count=("y_prob", "size"),
        responders=("y_true", "sum"),
    )
    st["non_responders"] = st["count"] - st["responders"]

    perfect = df.sort_values("y_true", ascending=False).copy()
    perfect["segment"] = np.linspace(1, n_segments + 1, len(perfect), endpoint=False, dtype=int)
    perfect_counts = perfect.groupby("segment")["y_true"].sum()

    st["expected_responders_random"] = df["y_true"].sum() / n_segments
    st["expected_responders_perfect"] = perfect_counts.reindex(st.index).fillna(0).to_numpy()
    st["response_rate"] = st["responders"] * 100 / st["count"]

    st["cumulative_count"] = st["count"].cumsum()
    st["cumulative_responders"] = st["responders"].cumsum()
    st["cumulative_perfect_responders"] = st["expected_responders_perfect"].cumsum()
    st["cumulative_non_responders"] = st["non_responders"].cumsum()

    total_count = st["count"].sum()
    total_responders = st["responders"].sum()
    total_perfect = st["expected_responders_perfect"].sum()
    total_non_responders = st["non_responders"].sum()
    random_rate = total_responders * 100 / total_count if total_count else np.nan

    st["cumulative_responders_topdown"] = st["responders"][::-1].cumsum()[::-1]
    st["cumulative_count_topdown"] = st["count"][::-1].cumsum()[::-1]
    st["cumulative_perfect_responders_topdown"] = st["expected_responders_perfect"][::-1].cumsum()[::-1]
    st["cumulative_count_percentage"] = st["cumulative_count"] * 100 / total_count if total_count else np.nan
    st["cumulative_count_percentage_topdown"] = (
        st["cumulative_count_topdown"] * 100 / total_count if total_count else np.nan
    )
    st["cumulative_responders_percentage"] = (
        st["cumulative_responders"] * 100 / total_responders if total_responders else np.nan
    )
    count_topdown = st["cumulative_count_topdown"]
    st["cumulative_responders_rate"] = np.where(
        count_topdown > 0, st["cumulative_responders_topdown"] * 100 / count_topdown, np.nan
    )
    st["cumulative_perfect_responders_rate"] = np.where(
        count_topdown > 0, st["cumulative_perfect_responders_topdown"] * 100 / count_topdown, np.nan
    )
    st["cumulative_random_responders_rate"] = random_rate

    st["cumulative_perfect_responders_percentage"] = (
        st["cumulative_perfect_responders"] * 100 / total_perfect if total_perfect else np.nan
    )
    st["cumulative_non_responders_percentage"] = (
        st["cumulative_non_responders"] * 100 / total_non_responders if total_non_responders else np.nan
    )

    st["ks_statistic"] = st["cumulative_responders_percentage"] - st["cumulative_non_responders_percentage"]
    st["lift"] = st["cumulative_responders_percentage"] / st["cumulative_count_percentage"]
    st["segment_default_rate"] = np.where(st["count"] > 0, st["responders"] * 100 / st["count"], np.nan)
    st["segment_default_rate_random"] = random_rate

    numeric_cols = [
        "probability_min",
        "probability_max",
        "probability_average",
        "response_rate",
        "cumulative_count_percentage",
        "cumulative_count_percentage_topdown",
        "cumulative_responders_percentage",
        "cumulative_perfect_responders_percentage",
        "cumulative_non_responders_percentage",
        "ks_statistic",
        "lift",
        "segment_default_rate",
        "segment_default_rate_random",
        "cumulative_responders_rate",
        "cumulative_random_responders_rate",
        "cumulative_perfect_responders_rate",
    ]
    st[numeric_cols] = st[numeric_cols].round(round_decimal)

    st = st.reset_index()

    return st


def _get_segment_tables(y_true, y_prob, n_segments=20, round_decimal=3, names=None):
    """Return a dict of segment tables keyed by model name."""
    y_true_array = np.asarray(y_true)
    y_prob_dict = _normalize_y_prob(y_prob, names=names)

    tables = {}
    for name, prob_values in y_prob_dict.items():
        prob_array = np.asarray(prob_values)
        if len(prob_array) != len(y_true_array):
            raise ValueError(
                f"y_prob for '{name}' has length {len(prob_array)} but y_true has length {len(y_true_array)}."
            )
        tables[name] = _segment_table_single(
            y_true_array, prob_array, n_segments=n_segments, round_decimal=round_decimal
        )

    return tables


def segment_table(y_true, y_prob, n_segments=20, labels=True, round_decimal=3, names=None):
    """
    Build segment tables for binary classifier scores.

    Accepts a single vector/Series/array or multiple as dict, DataFrame columns, or list/tuple.
    Returns a DataFrame for a single model or a dict of DataFrames for multiple models.
    """
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, round_decimal=round_decimal, names=names)

    if labels:
        print_labels()

    return next(iter(tables.values())) if len(tables) == 1 else tables


def plot_lift(
    y_true, y_prob, n_segments=20, title="Lift Plot", names=None, **kwargs
):
    """Plot cumulative lift by segment; supports multiple models."""
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, names=names)

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)
    max_percentage = max(dt["cumulative_count_percentage"].max() for dt in tables.values())

    for name, dt in tables.items():
        plt.plot(dt["cumulative_count_percentage"], dt["lift"], marker="o", label=name)

    plt.plot([0, max_percentage], [1, 1], "k--", marker="o", label="Random")
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Population (%)", fontsize=text_fontsize)
    plt.ylabel("Lift", fontsize=text_fontsize)
    plt.legend(fontsize=text_fontsize)
    plt.grid(True)


def plot_lift_segment_wise(
    y_true, y_prob, n_segments=20, title="Segment-wise Lift Plot", names=None, **kwargs
):
    """Plot lift per segment; supports multiple models."""
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, names=names)
    max_percentage = max(dt["cumulative_count_percentage"].max() for dt in tables.values())

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)

    for name, dt in tables.items():
        lift_by_segment = dt["responders"] / dt["expected_responders_random"]
        plt.plot(dt["cumulative_count_percentage"], lift_by_segment, marker="o", label=name)

    plt.plot([0, max_percentage], [1, 1], "k--", marker="o", label="Random")
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Population (%)", fontsize=text_fontsize)
    plt.ylabel("Lift @ Segment", fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)


def plot_cumulative_default_rate(
    y_true,
    y_prob,
    n_segments=20,
    title="Cumulative Default Rate by Approved Population",
    names=None,
    **kwargs
):
    """Plot cumulative default rate by approved population."""
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, names=names)

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)

    for name, dt in tables.items():
        x_values = dt["cumulative_count_percentage_topdown"].values
        plt.plot(
            x_values,
            dt["cumulative_responders_rate"].values,
            marker="o",
            label=name,
        )

    first_dt = next(iter(tables.values()))
    x_base = first_dt["cumulative_count_percentage_topdown"].values
    plt.plot(
        x_base,
        first_dt["cumulative_random_responders_rate"].values,
        linestyle="--",
        color="gray",
        label="Random",
    )
    plt.plot(
        x_base,
        first_dt["cumulative_perfect_responders_rate"].values,
        linestyle="-.",
        color="green",
        label="Perfect",
    )

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Population (%)", fontsize=text_fontsize)
    plt.ylabel("Default Rate (%)", fontsize=text_fontsize)
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_segment_default_rate(
    y_true,
    y_prob,
    n_segments=20,
    title="Segment-wise Default Rate by Approved Population",
    names=None,
    **kwargs
):
    """Plot segment-wise default rate by approved population."""
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, names=names)

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)
    max_percentage = max(dt["cumulative_count_percentage_topdown"].max() for dt in tables.values())
    min_percentage = min(dt["cumulative_count_percentage_topdown"].min() for dt in tables.values())

    for name, dt in tables.items():
        plt.plot(
            dt["cumulative_count_percentage_topdown"],
            dt["segment_default_rate"],
            marker="o",
            label=name,
        )

    first_dt = next(iter(tables.values()))
    random_rate = first_dt["segment_default_rate_random"].iloc[0] if hasattr(first_dt, "iloc") else first_dt["segment_default_rate_random"]
    plt.plot([min_percentage, max_percentage], [random_rate, random_rate], linestyle="--", color="gray", label="Random")

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Approved Population (%)", fontsize=text_fontsize)
    plt.ylabel("Default Rate (%)", fontsize=text_fontsize)
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_cumulative_gain(
    y_true,
    y_prob,
    n_segments=20,
    title="Cumulative Gain Plot",
    names=None,
    **kwargs
):
    """Plot cumulative gain curves for one or more models."""
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, names=names)

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)
    max_percentage = max(dt["cumulative_count_percentage"].max() for dt in tables.values())

    for name, dt in tables.items():
        plt.plot(
            np.append(0, dt["cumulative_count_percentage"].values),
            np.append(0, dt["cumulative_responders_percentage"].values),
            marker="o",
            label=name,
        )

    first_dt = next(iter(tables.values()))
    plt.plot(
        np.append(0, first_dt["cumulative_count_percentage"].values),
        np.append(0, first_dt["cumulative_perfect_responders_percentage"].values),
        "c--",
        label="Perfect",
    )
    plt.plot([0, max_percentage], [0, 100], "k--", marker="o", label="Random")
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Population (%)", fontsize=text_fontsize)
    plt.ylabel("% Responders", fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)


def plot_ks_statistic(
    y_true,
    y_prob,
    n_segments=20,
    title="KS Statistic Plot",
    names=None,
    **kwargs
):
    """Plot KS statistic curves for responders and non-responders; supports multiple models."""
    tables = _get_segment_tables(y_true, y_prob, n_segments=n_segments, names=names)

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)

    for name, dt in tables.items():
        x_values = np.append(0, dt["cumulative_count_percentage"].values)
        plt.plot(
            x_values,
            np.append(0, dt["cumulative_responders_percentage"].values),
            marker="o",
            label=f"{name} - Responders",
        )

        ks_value = dt["ks_statistic"].max()
        ks_segment = dt.loc[dt["ks_statistic"] == ks_value, "segment"].values[0]
        ks_percentage = dt.loc[dt["segment"] == ks_segment, "cumulative_count_percentage"].values[0]
        plt.plot(
            [ks_percentage, ks_percentage],
            [
                dt.loc[dt["segment"] == ks_segment, "cumulative_responders_percentage"].values[0],
                dt.loc[dt["segment"] == ks_segment, "cumulative_non_responders_percentage"].values[0],
            ],
            "g--",
            marker="o",
            label=f"{name} KS: {ks_value} (Seg. {ks_segment})",
        )
    
    plt.plot(
        x_values,
        np.append(0, dt["cumulative_non_responders_percentage"].values),
        marker="o",
        label=f"Non-Responders",
    )

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Population (%)", fontsize=text_fontsize)
    plt.ylabel("% Responders", fontsize=text_fontsize)
    plt.legend()
    plt.grid(True)


def performance_report(
    y_true,
    y_prob,
    n_segments=20,
    labels=True,
    plot_style=None,
    round_decimal=3,
    names=None,
    **kwargs
):
    """Plot lift, gain, and KS curves alongside the segment table."""
    st = segment_table(
        y_true, y_prob, n_segments=n_segments, labels=labels, round_decimal=round_decimal, names=names
    )

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)
    figsize = kwargs.get("figsize", (16, 10))

    if plot_style:
        plt.style.use(plot_style)

    plt.figure(figsize=figsize)

    plt.subplot(2, 2, 1)
    plot_lift(y_true, y_prob, n_segments=n_segments, names=names)

    plt.subplot(2, 2, 2)
    plot_lift_segment_wise(y_true, y_prob, n_segments=n_segments, names=names)

    plt.subplot(2, 2, 3)
    plot_cumulative_gain(y_true, y_prob, n_segments=n_segments, names=names)

    plt.subplot(2, 2, 4)
    plot_ks_statistic(y_true, y_prob, n_segments=n_segments, names=names)

    return st


def default_report(
    y_true,
    y_prob,
    n_segments=20,
    labels=True,
    plot_style=None,
    round_decimal=3,
    names=None,
    **kwargs
):
    """Plot cumulative and segment default rate charts."""
    st = segment_table(
        y_true, y_prob, n_segments=n_segments, labels=labels, round_decimal=round_decimal, names=names
    )

    title_fontsize = kwargs.get("title_fontsize", 14)
    text_fontsize = kwargs.get("text_fontsize", 10)
    figsize = kwargs.get("figsize", (16, 5))

    if plot_style:
        plt.style.use(plot_style)

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plot_cumulative_default_rate(
        y_true,
        y_prob,
        n_segments=n_segments,
        names=names,
        title_fontsize=title_fontsize,
        text_fontsize=text_fontsize,
    )

    plt.subplot(1, 2, 2)
    plot_segment_default_rate(
        y_true,
        y_prob,
        n_segments=n_segments,
        names=names,
        title_fontsize=title_fontsize,
        text_fontsize=text_fontsize,
    )

    return st
