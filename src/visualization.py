from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR

def plot_class_distribution(counts, save_path=None):
    out = Path(save_path or (FIGURES_DIR / "class_distribution.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def plot_confusion_matrix_heatmap(cm, classes, save_path=None):
    out = Path(save_path or (FIGURES_DIR / "evaluation_confusion_matrix.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out
