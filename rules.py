"""Extract random-forest rules and calculate their feature-level statistics."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

TREE_UNDEFINED = -2


def extract_rules(pipeline):
    model = pipeline.named_steps["classifier"]
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    rows = []
    for tree_index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        def visit(node, intervals, categorical):
            feature_index = tree.feature[node]
            if feature_index == TREE_UNDEFINED:
                constraints = {**{names[i]: value for i, value in intervals.items()}, **{
                    name: value for name, value in categorical.items() if value in (0, 1)}}
                rows.append({"Tree": tree_index, "Class": int(model.classes_[np.argmax(tree.value[node][0])]),
                             "Constraints": constraints})
                return
            name, threshold = names[feature_index], float(tree.threshold[node])
            if name.startswith("cat__"):
                left, right = dict(categorical), dict(categorical)
                left[name], right[name] = 0, 1
                visit(tree.children_left[node], dict(intervals), left)
                visit(tree.children_right[node], dict(intervals), right)
            else:
                left, right = dict(intervals), dict(intervals)
                lo, hi = left.get(feature_index, (-np.inf, np.inf)); left[feature_index] = (lo, min(hi, threshold))
                lo, hi = right.get(feature_index, (-np.inf, np.inf)); right[feature_index] = (max(lo, np.nextafter(threshold, np.inf)), hi)
                visit(tree.children_left[node], left, dict(categorical))
                visit(tree.children_right[node], right, dict(categorical))
        visit(0, {}, {})
    return pd.DataFrame(rows)


def rule_statistics(rules: pd.DataFrame, metadata: dict, encoded_names: list[str], edges_by_feature: dict):
    """Return rule counts per original feature/category/bin/class."""
    index = {name: i for i, name in enumerate(encoded_names)}
    counts = defaultdict(int); distinct = defaultdict(set)
    for rule_id, rule in rules.iterrows():
        for feature, info in metadata.items():
            for column in info.encoded_columns:
                constraint = rule.Constraints.get(column)
                if constraint is None:
                    continue
                distinct[(feature, info.label_for(column), int(rule.Class))].add(rule_id)
                if info.kind == "categorical":
                    counts[(feature, info.label_for(column), int(constraint), int(rule.Class))] += 1
                else:
                    lo, hi = constraint
                    for bin_id in range(len(edges_by_feature[feature]) - 1):
                        left, right = edges_by_feature[feature][bin_id:bin_id + 2]
                        if not (hi < left or lo > right):
                            counts[(feature, None, bin_id, int(rule.Class))] += 1
    return counts, distinct
