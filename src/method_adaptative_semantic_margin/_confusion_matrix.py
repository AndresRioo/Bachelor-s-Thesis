from sklearn.manifold import TSNE
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(cm, labels_sorted, idx_to_class, name, out_dir="analysis"):
    """
    Guarda una matriz de confusión como imagen PNG.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        xticklabels=[idx_to_class[i] for i in labels_sorted],
        yticklabels=[idx_to_class[i] for i in labels_sorted],
        cmap="viridis",
        square=True
    )
    plt.xlabel("Predicted")
    plt.ylabel("GT")
    plt.title(name)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    plt.close()


def compute_confusion_matrices(all_gt, all_pred_proto, all_pred_sem, all_pred_mix, labels_sorted):
    """
    Calcula matrices de confusión (raw y normalizada) para proto, sem y mix.
    """
    cm_proto = confusion_matrix(all_gt, all_pred_proto, labels=labels_sorted)
    cm_sem = confusion_matrix(all_gt, all_pred_sem, labels=labels_sorted)
    cm_mix = confusion_matrix(all_gt, all_pred_mix, labels=labels_sorted)

    row_sums = cm_mix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_mix_norm = cm_mix / row_sums

    return cm_proto, cm_sem, cm_mix, cm_mix_norm