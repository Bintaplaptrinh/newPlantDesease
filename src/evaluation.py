# các hàm đánh giá model: confusion matrix, classification report, roc micro (ovr)

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


def maybe_load_model(path: str, fallback_model: torch.nn.Module, device: str):
    # load model đã lưu nếu có, nếu không thì dùng fallback_model
    if os.path.exists(path):
        m = torch.load(path, weights_only=False, map_location=device)
        m = m.to(device)
        m.eval()
        return m
    fallback_model.eval()
    return fallback_model


@torch.no_grad()
def predict_logits_and_targets(model: torch.nn.Module, loader, device: str):
    # chạy infer để lấy logits và gom targets
    model.eval()
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        all_logits.append(out.detach().float().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    return logits, targets


def per_class_accuracy_from_cm(cm: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    # tính accuracy theo từng class từ confusion matrix
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)
    return pd.DataFrame({"class": class_names, "support": support, "correct": correct, "accuracy": acc})


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, normalize: bool = True):
    # tạo figure confusion matrix (không gọi plt.show)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig


def evaluate_one_model(name: str, model: torch.nn.Module, loader, device: str, class_names: List[str], num_classes: int, return_figures: bool = True):
    # tính metrics để export cho web (không print, không plt.show)
    logits, y_true = predict_logits_and_targets(model, loader, device)
    y_prob = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = logits.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    overall_accuracy = float((y_pred == y_true).mean())
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    df_acc = per_class_accuracy_from_cm(cm, class_names).sort_values("accuracy", ascending=True)

    figures = {}
    if return_figures:
        figures["confusion_matrix"] = plot_confusion_matrix(
            cm,
            class_names,
            title=f"{name} - Confusion Matrix (normalized)",
            normalize=True,
        )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "overall_accuracy": overall_accuracy,
        "classification_report": report_dict,
        "per_class_accuracy": df_acc,
        "confusion_matrix": cm,
        "figures": figures,
    }


def plot_roc_micro_ovr(results: Dict[str, dict], num_classes: int, title: str = "ROC Curve Comparison (micro-average, OvR)"):
    # tạo figure roc (không gọi plt.show)
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs: Dict[str, Dict[str, float]] = {}

    for name, pack in results.items():
        y_true = pack["y_true"]
        y_prob = pack["y_prob"]
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        try:
            auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc_macro = None

        label = f"{name} (micro AUC={auc_micro:.4f}" + (f", macro AUC={auc_macro:.4f}" if auc_macro is not None else "") + ")"
        aucs[name] = {"micro": float(auc_micro)}
        if auc_macro is not None:
            aucs[name]["macro"] = float(auc_macro)

        ax.plot(fpr_micro, tpr_micro, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return {"figure": fig, "aucs": aucs}
