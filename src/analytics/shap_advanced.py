from __future__ import annotations

from pathlib import Path
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _prepare_shap_inputs(model, X_sample: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """Prepare the classifier and transformed feature matrix for SHAP."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed = preprocessor.transform(X_sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        feature_names = preprocessor.get_feature_names_out()
        X_for_shap = pd.DataFrame(transformed, columns=feature_names, index=X_sample.index)
        return classifier, X_for_shap

    return model, X_sample


def run_shap_analysis(model, X_sample: pd.DataFrame, save_dir: str | Path) -> None:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier, X_for_shap = _prepare_shap_inputs(model, X_sample)
    # Transform data using pipeline
    X_transformed = model.named_steps["preprocessor"].transform(X_sample)

    # Use TreeExplainer on classifier
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_for_shap, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "08_shap_summary.png", dpi=150)
    plt.close()

    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "09_shap_bar.png", dpi=150)
    plt.close()

    probs = model.predict_proba(X_sample)[:, 1]
    high_idx = int(np.argmax(probs))
    low_idx = int(np.argmin(probs))
    mid_idx = int(np.argmin(np.abs(probs - 0.5)))

    for idx, label in ((high_idx, "high"), (low_idx, "low"), (mid_idx, "mid")):
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_{label}.png", dpi=150)
        plt.close()
