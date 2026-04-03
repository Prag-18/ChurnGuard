from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(name, model, X_test, y_test, save_dir):
    """Evaluate a model and return metrics plus predicted probabilities."""
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(output_dir / f"{name}_confusion_matrix.png")
    plt.close()

    return metrics, y_proba


def plot_roc_curves(models_dict, y_test, save_path):
    plt.figure()

    for name, (model, X_test) in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=name)

    plt.legend()
    plt.title("ROC Curves")
    plt.savefig(save_path)
    plt.close()


def plot_pr_curves(models_dict, y_test, save_path):
    plt.figure()

    for name, (model, X_test) in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=name)

    plt.legend()
    plt.title("Precision-Recall Curves")
    plt.savefig(save_path)
    plt.close()
