"""Reusable, explainer-agnostic plotting functions."""
from __future__ import annotations

import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def fig_to_src(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def numeric_edges(data, bins: int):
    answer = {}
    index = {name: i for i, name in enumerate(data.encoded_feature_names)}
    for name, info in data.metadata.items():
        if info.kind != "numerical":
            continue
        values = np.concatenate((data.X_train_encoded[:, index[info.encoded_columns[0]]],
                                 data.X_test_encoded[:, index[info.encoded_columns[0]]]))
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if low == high:
            low, high = low - .5, high + .5
        answer[name] = np.linspace(low, high, int(bins) + 1)
    return answer


def global_limit(explanation_frames):
    values = [frame.Importance.abs().max() for frame in explanation_frames.values() if not frame.empty]
    return max(values, default=1.0) or 1.0


def _overlay_rules(axis, matrix, feature, category_keys, cls, rule_counts):
    maximum = max((rule_counts.get((feature, category, b, cls), 0)
                   for category in category_keys for b in range(matrix.shape[1])), default=0)
    if not maximum:
        return
    for row, category in enumerate(category_keys):
        for col in range(matrix.shape[1]):
            count = rule_counts.get((feature, category, col, cls), 0)
            if count:
                # Leave most of the coloured cell exposed; the marker remains comparable.
                side = .55 * count / maximum
                axis.add_patch(patches.Rectangle((col - side / 2, row - side / 2), side, side,
                    linewidth=.7, edgecolor="black", facecolor="none"))


def plot_feature_comparison(feature, metadata, explanations, rule_counts, cls, vlim):
    """Rule histogram plus TreeSHAP, KernelSHAP, and LIME using the same colour scale."""
    info = metadata[feature]
    categories = [info.label_for(c) for c in info.encoded_columns] if info.kind == "categorical" else [feature]
    rule_categories = categories if info.kind == "categorical" else [None]
    bins = 2 if info.kind == "categorical" else max(frame.Bin.max() for frame in explanations.values()) + 1
    matrices = {}
    for method, frame in explanations.items():
        selected = frame[(frame.Feature == feature) & (frame.Class == cls)]
        matrix = np.zeros((len(categories), bins))
        for row, category in enumerate(categories):
            records = selected[selected.Category.eq(category)] if info.kind == "categorical" else selected
            for _, record in records.iterrows():
                matrix[row, int(record.Bin)] = record.Importance
        matrices[method] = matrix
    fig, axes = plt.subplots(1, 4, figsize=(17, max(2.8, .42 * len(categories))), gridspec_kw={"width_ratios": [1, 2.5, 2.5, 2.5]}, constrained_layout=True)
    bar_values = [sum(rule_counts.get((feature, category, bin_id, cls), 0) for bin_id in range(bins)) for category in rule_categories]
    axes[0].barh(np.arange(len(categories)), bar_values, color="#b8b8b8", edgecolor="#666", linewidth=.5)
    axes[0].set(yticks=np.arange(len(categories)), yticklabels=categories, xlabel="Rule count", title="Rules")
    axes[0].tick_params(axis="y", pad=8)
    image = None
    for axis, (method, matrix) in zip(axes[1:], matrices.items()):
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        _overlay_rules(axis, matrix, feature, rule_categories, cls, rule_counts)
        axis.set(title=method, xlabel="Category state (0 / 1)" if info.kind == "categorical" else "Value bin",
                 yticks=np.arange(len(categories)), yticklabels=[])
        axis.set_xticks(np.arange(bins))
    fig.colorbar(image, ax=axes[1:], shrink=.88, label="Mean signed importance")
    return fig


def plot_numeric_features_comparison(metadata, explanations, rule_counts, cls, vlim):
    """Display every numeric feature as rows in the same method comparison."""
    features = [name for name, info in metadata.items() if info.kind == "numerical"]
    bins = max(int(frame.Bin.max()) for frame in explanations.values()) + 1
    matrices = {}
    for method, frame in explanations.items():
        matrix = np.zeros((len(features), bins))
        for row, feature in enumerate(features):
            selected = frame[(frame.Feature == feature) & (frame.Class == cls)]
            for _, record in selected.iterrows():
                matrix[row, int(record.Bin)] = record.Importance
        matrices[method] = matrix
    rule_matrix = np.asarray([[rule_counts.get((feature, None, bin_id, cls), 0) for bin_id in range(bins)] for feature in features])
    rule_totals = rule_matrix.sum(axis=1)
    fig, axes = plt.subplots(1, 4, figsize=(18, max(4.5, .6 * len(features))),
                             gridspec_kw={"width_ratios": [1, 2.8, 2.8, 2.8]}, constrained_layout=True)
    axes[0].barh(np.arange(len(features)), rule_totals, color="#b8b8b8", edgecolor="#666", linewidth=.5)
    axes[0].set(yticks=np.arange(len(features)), yticklabels=[name.replace("_", " ") for name in features], xlabel="Rule count", title="Rules")
    axes[0].tick_params(axis="y", pad=8)
    image = None
    maximum = rule_matrix.max()
    for axis, (method, matrix) in zip(axes[1:], matrices.items()):
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        if maximum:
            for row, feature in enumerate(features):
                for col in range(bins):
                    count = rule_matrix[row, col]
                    if count:
                        side = .55 * count / maximum
                        axis.add_patch(patches.Rectangle((col - side / 2, row - side / 2), side, side,
                                       linewidth=.7, edgecolor="black", facecolor="none"))
        axis.set(title=method, xlabel="Value bin", yticks=np.arange(len(features)), yticklabels=[])
        axis.set_xticks(np.arange(bins))
    fig.colorbar(image, ax=axes[1:], shrink=.88, label="Mean signed importance")
    return fig


def plot_overview(explanations, metadata, cls, vlim):
    """One column per explainer and one row per original feature."""
    features = list(metadata)
    fig, axes = plt.subplots(1, len(explanations), figsize=(5 * len(explanations), max(4, .38 * len(features))), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, (method, frame) in zip(axes, explanations.items()):
        values = []
        for feature in features:
            subset = frame[(frame.Feature == feature) & (frame.Class == cls)]
            values.append(float(subset.Importance.abs().mean()) if not subset.empty else 0.)
        image = axis.imshow(np.asarray(values)[:, None], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        axis.set(title=method, xticks=[], yticks=np.arange(len(features)), yticklabels=features if axis is axes[0] else [])
        if axis is axes[0]:
            axis.tick_params(axis="y", pad=8)
    fig.colorbar(image, ax=axes, shrink=.85, label="Mean absolute importance")
    return fig
