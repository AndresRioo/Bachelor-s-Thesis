

from sklearn.manifold import TSNE
from analysis.common_utils import safe_pca

from sklearn.preprocessing import normalize
import plotly.express as px

"""
Color palette used for all t-SNE visualizations.

Alphabet provides 26 visually distinct colors,
which is sufficient for all evaluated datasets.
"""
TSNE_COLORS = px.colors.qualitative.Alphabet

# ├── tsne/
# │   ├── embeddings/
# │   ├── prototype_visualization/
# │   ├── prototype_distribution/
# │   └── query_embeddings/



def generate_tsne_plot(
    X,
    labels,
    save_path,
    title,
    symbols=None,
    perplexity=30,
    learning_rate=200
):
    """
    Generate and save a complete t-SNE visualization.

    This helper centralizes the full t-SNE pipeline used
    throughout the analysis section.

    Processing pipeline
    -------------------
    1. L2-normalize input features.
    2. Apply PCA dimensionality reduction.
    3. Compute a 2D t-SNE projection.
    4. Generate an interactive Plotly visualization.
    5. Save the result as an HTML file.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    labels : list[str]
        Class label associated with each sample.

    save_path : str
        Output HTML path.

    title : str
        Figure title.

    symbols : list[str], optional
        Marker type assigned to each sample.

    perplexity : int
        t-SNE perplexity.

    learning_rate : int
        t-SNE learning rate.

    Returns
    -------
    None
    """
    
    print("[TSNE] normalize")

    X = normalize(X)

    print("[TSNE] projection")

    X_2d = compute_tsne_projection(
        X,
        perplexity=perplexity,
        learning_rate=learning_rate
    )

    print("[TSNE] scatter")

    save_tsne_scatter(
        X_2d,
        labels,
        title,
        save_path,
        symbols=symbols
    )

    print("[TSNE] done")


def compute_tsne_projection(
    X,
    perplexity=30,
    learning_rate=200,
    random_state=0
):
    """
    Compute a 2D t-SNE projection.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.

    perplexity : float
        Controls local neighbourhood size.

    learning_rate : float
        Optimization learning rate.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Two-dimensional embedding.
    """

    print("Before PCA:", X.shape)

    X = safe_pca(X)

    print("After PCA:", X.shape)


    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=0
    )

    print("[TSNE] fit_transform start")
    result = tsne.fit_transform(X)
    print("[TSNE] fit_transform done")

    return result



def save_tsne_scatter(
    X_2d,
    labels,
    title,
    save_path,
    symbols=None
):
    """
    Save an interactive Plotly t-SNE visualization.

    Parameters
    ----------
    X_2d : np.ndarray
        2D t-SNE coordinates.

    labels : list[str]
        Class labels.

    title : str
        Plot title.

    save_path : str
        HTML output file.

    symbols : list[str], optional
        Marker type for each point.

    Returns
    -------
    None
    """

    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=labels,
        symbol=symbols,
        title=title,
        width=1000,
        height=800,
        color_discrete_sequence=TSNE_COLORS
    )

    fig.update_traces(marker=dict(size=5))

    fig.write_html(save_path)
