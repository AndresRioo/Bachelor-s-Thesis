from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os 
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# confusion_matrix/
# │
# ├── matrices/
# │   ├── proto_confusion.html
# │   ├── gen_confusion.html
# │   └── mix_confusion.html
# │
# ├── matrices_normalized/
# │   ├── proto_confusion_norm.html
# │   ├── gen_confusion_norm.html
# │   └── mix_confusion_norm.html
# │
# ├── matrices_difference/
# │   ├── gen_minus_proto.html
# │   ├── mix_minus_proto.html
# │   └── mix_minus_gen.html
# │
# ├── confusion_matrices.xlsx
# └── statistics_analysis.xlsx


def save_interactive_cm(
    cm,
    class_names,
    filename,
    title,
    save_dir
):
    """
    Save an interactive confusion matrix as HTML.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.

    class_names : list[str]
        Class labels.

    filename : str
        Output filename.

    title : str
        Figure title.

    save_dir : str
        Output directory.

    Returns
    -------
    None
    """

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale="Viridis",
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        width=1000,
        height=1000
    )

    fig.write_html(os.path.join(save_dir, filename))


def compute_metrics(
    y_true,
    y_pred
):
    """
    Compute global classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict
        Accuracy, precision, recall and F1
        using both macro and micro averaging.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro")
    }


def compute_ova(cm, class_names):
    """
    Compute One-vs-All metrics for each class.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.

    class_names : list[str]
        Class labels.

    Returns
    -------
    pandas.DataFrame
        Precision, recall, specificity and F1
        for every class.
    """

    results = []
    total = cm.sum()

    for i, cls in enumerate(class_names):

        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = total - TP - FP - FN

        precision = TP / (TP + FP) if TP + FP else 0
        recall = TP / (TP + FN) if TP + FN else 0
        specificity = TN / (TN + FP) if TN + FP else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        results.append({
            "class": cls,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1
        })

    return pd.DataFrame(results)


def top_confusions(cm, class_names, k=20):
    """
    Return the most frequent off-diagonal confusions.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.

    class_names : list[str]
        Class labels.

    k : int
        Number of confusion pairs to return.

    Returns
    -------
    pandas.DataFrame
        Top-k class confusions sorted by frequency.
    """

    confusions = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append(
                    (class_names[i], class_names[j], cm[i, j])
                )

    confusions.sort(key=lambda x: x[2], reverse=True)

    return pd.DataFrame(
        confusions[:k],
        columns=["True_Class", "Predicted_Class", "Count"]
    )


def per_class_accuracy(
    cm,
    class_names
):
    """
    Compute per-class classification accuracy.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.

    class_names : list[str]
        Class labels.

    Returns
    -------
    pandas.DataFrame
        Accuracy for every class.
    """
    acc = cm.diagonal() / cm.sum(axis=1)

    return pd.DataFrame({
        "class": class_names,
        "accuracy": acc
    })

