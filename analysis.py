"""Build the model analysis shared by the Dash and static web frontends."""
from __future__ import annotations

from explainers import compute_kernel_shap, compute_lime, compute_tree_shap
from preprocessing import train_model
from rules import extract_rules


def build_analysis():
    """Return the trained data, extracted rules, and all explanation methods."""
    data = train_model()
    rules = extract_rules(data.pipeline)
    explanations = {"TreeSHAP": compute_tree_shap(data)}
    for name, factory in (("KernelSHAP", compute_kernel_shap), ("LIME", compute_lime)):
        try:
            explanations[name] = factory(data)
        except Exception as error:
            raise RuntimeError(f"{name} could not be computed; all three methods are required.") from error
    return data, rules, explanations
