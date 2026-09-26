"""Run the German Credit XAI comparison dashboard."""
from preprocessing import train_model
from rules import extract_rules
from explainers import compute_kernel_shap, compute_lime, compute_tree_shap
from dashboard import create_app


def main():
    data = train_model()
    rules = extract_rules(data.pipeline)
    explanations = {"TreeSHAP": compute_tree_shap(data)}
    for name, factory in (("KernelSHAP", compute_kernel_shap), ("LIME", compute_lime)):
        try:
            explanations[name] = factory(data)
        except Exception as error:
            raise RuntimeError(f"{name} could not be computed; all three methods are required.") from error
    create_app(data, rules, explanations).run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
