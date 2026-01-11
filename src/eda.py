from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def count_per_class_from_imagefolder_samples(samples: List[Tuple[str, int]], class_names: List[str]) -> pd.DataFrame:
    # Counts images per class
    labels = [label for _, label in samples]
    labels_count = Counter(labels)
    class_count: Dict[str, int] = {class_names[k]: v for k, v in labels_count.items()}
    df_count = pd.DataFrame({"Class": list(class_count.keys()), "num_classes": list(class_count.values())})
    df_count = df_count.sort_values(by="num_classes", ascending=False)
    return df_count


def plot_class_distribution(df_count: pd.DataFrame, title: str = "Biểu đồ số lượng ảnh theo từng lớp"):
    # Build class distribution figure (no plt.show)
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(df_count["Class"], df_count["num_classes"])

    ax.set_title(title)
    ax.set_xlabel("Lớp")
    ax.set_ylabel("Số lượng ảnh")
    ax.tick_params(axis="x", rotation=45)

    for b, v in zip(bars, df_count["num_classes"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{int(v)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def is_over_bright(img_path: str, thresh: float = 0.85):
    # Simple brightness check on HSV-V channel
    img = cv2.imread(img_path)
    if img is None:
        return False, 0.0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2] / 255.0
    mean_v = float(v.mean())
    return mean_v > thresh, mean_v


def over_bright_report(samples: List[Tuple[str, int]], class_names: List[str], class_count: Dict[str, int], thresh: float = 0.85) -> pd.DataFrame:
    # Computes over-bright ratio per class
    over_count = Counter()

    for img_path, label in tqdm(samples):
        is_over, _ = is_over_bright(img_path, thresh=thresh)
        if is_over:
            class_name = class_names[label]
            over_count[class_name] += 1

    df_over = pd.DataFrame.from_dict(
        {
            "Class": list(over_count.keys()),
            "Total Images": [class_count[c] for c in over_count],
            "Over Bright Images": [over_count[c] for c in over_count],
        }
    )

    df_over["Ratio"] = df_over["Over Bright Images"] / df_over["Total Images"]
    df_over = df_over.sort_values(by="Ratio", ascending=False)
    return df_over


def plot_over_bright_ratio(df_over: pd.DataFrame, title: str = "Tỷ lệ ảnh quá sáng theo từng lớp"):
    # Build over-bright ratio figure (no plt.show)
    x = df_over["Class"]
    y = df_over["Ratio"]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(x, y)

    ax.set_title(title)
    ax.set_xlabel("Lớp")
    ax.set_ylabel("Tỷ lệ ảnh quá sáng")
    ax.tick_params(axis="x", rotation=45)

    for bar, over_n, total_n, ratio in zip(
        bars,
        df_over["Over Bright Images"],
        df_over["Total Images"],
        df_over["Ratio"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{over_n}/{total_n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig
