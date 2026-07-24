
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram


def save_dendrogram(
    linkage_matrix,
    labels,
    title,
    save_path
):
    """
    Save a hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Output of scipy linkage().

    labels : list[str]
        Class names.

    title : str
        Figure title.

    save_path : str
        Output image path.

    Returns
    -------
    None
    """

    plt.figure(figsize=(18, 8))

    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def build_mutual_confusion_distance(
    cm_norm
):
    """
    Build a distance matrix from mutual confusion.

    Two classes are considered similar when they are
    frequently confused in both directions.

    similarity(i,j) =
        cm(i,j) + cm(j,i)

    Parameters
    ----------
    cm_norm : np.ndarray
        Row-normalized confusion matrix.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix.
    """

    cm = cm_norm.copy()

    np.fill_diagonal(cm, 0)

    mutual_similarity = cm + cm.T

    if mutual_similarity.max() > 0:
        mutual_similarity = (
            mutual_similarity /
            mutual_similarity.max()
        )

    distance_matrix = 1.0 - mutual_similarity

    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix