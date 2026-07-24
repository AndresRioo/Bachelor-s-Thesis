from sklearn.decomposition import PCA

def safe_pca(X, max_components=50):
    """
    Applies PCA before t-SNE.

    Parameters
    ----------
    X : np.ndarray
        Input embedding matrix.

    max_components : int, optional
        Maximum number of PCA components.

    Returns
    -------
    np.ndarray
        PCA-transformed embeddings.

    Notes
    -----
    PCA is used to reduce dimensionality before
    applying t-SNE, improving stability and speed.
    """    
    n_components = min(max_components, X.shape[0], X.shape[1])
    return PCA(n_components=n_components).fit_transform(X)


