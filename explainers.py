"""Explanation methods.  Plotting code only consumes their shared result type."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap


@dataclass
class ExplanationResult:
    method: str
    values_by_class: dict[int, np.ndarray]


def _as_class_dict(values, classes):
    values = np.asarray(values) if not isinstance(values, list) else values
    if isinstance(values, list):
        return {int(c): np.asarray(values[i]) for i, c in enumerate(classes)}
    if values.ndim == 3 and values.shape[-1] == len(classes):
        return {int(c): values[:, :, i] for i, c in enumerate(classes)}
    if values.ndim == 2 and len(classes) == 2:
        return {int(classes[0]): -values, int(classes[1]): values}
    raise ValueError(f"Unexpected explanation shape: {values.shape}")


def compute_tree_shap(data) -> ExplanationResult:
    model = data.pipeline.named_steps["classifier"]
    background = shap.kmeans(data.X_train_encoded, min(50, len(data.X_train_encoded)))
    try:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output="probability")
    except ValueError:
        explainer = shap.TreeExplainer(model)
    return ExplanationResult("TreeSHAP", _as_class_dict(explainer.shap_values(data.X_test_encoded), data.classes))


def compute_kernel_shap(data, nsamples: int = 100) -> ExplanationResult:
    model = data.pipeline.named_steps["classifier"]
    background = shap.kmeans(data.X_train_encoded, min(50, len(data.X_train_encoded)))
    values = shap.KernelExplainer(model.predict_proba, background).shap_values(data.X_test_encoded, nsamples=nsamples)
    return ExplanationResult("KernelSHAP", _as_class_dict(values, data.classes))


def compute_lime(data, num_samples: int = 1000) -> ExplanationResult:
    from lime.lime_tabular import LimeTabularExplainer
    model = data.pipeline.named_steps["classifier"]
    explainer = LimeTabularExplainer(data.X_train_encoded, mode="classification", feature_names=data.encoded_feature_names,
                                     class_names=[str(c) for c in data.classes], discretize_continuous=True, random_state=42)
    result = {c: np.zeros_like(data.X_test_encoded, dtype=float) for c in data.classes}
    class_index = {c: i for i, c in enumerate(data.classes)}
    for sample_id, row in enumerate(data.X_test_encoded):
        predicted = int(data.predicted_classes[sample_id]); label = class_index[predicted]
        explanation = explainer.explain_instance(row, model.predict_proba, labels=[label], num_features=len(row), num_samples=num_samples)
        for column, value in explanation.as_map().get(label, []):
            result[predicted][sample_id, column] = value
    return ExplanationResult("LIME", result)


def aggregate_explanations(result, data, edges_by_feature):
    """Return the common Feature, Category, Class, Bin, Importance dataframe."""
    column_index = {name: i for i, name in enumerate(data.encoded_feature_names)}
    rows = []
    for feature, info in data.metadata.items():
        for column in info.encoded_columns:
            column_id = column_index[column]
            values = data.X_test_encoded[:, column_id]
            bins = (values > .5).astype(int) if info.kind == "categorical" else pd.cut(values, edges_by_feature[feature], labels=False, include_lowest=True)
            for cls, matrix in result.values_by_class.items():
                mask = (data.predicted_classes == cls) & ~pd.isna(bins)
                for bin_id in np.unique(np.asarray(bins)[mask]):
                    selected = mask & (np.asarray(bins) == bin_id)
                    rows.append({"Feature": feature, "Category": info.label_for(column) if info.kind == "categorical" else None,
                                 "Class": cls, "Bin": int(bin_id), "Importance": float(np.mean(matrix[selected, column_id]))})
    return pd.DataFrame(rows)
