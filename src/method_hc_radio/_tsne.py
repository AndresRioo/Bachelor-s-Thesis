from sklearn.manifold import TSNE
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler

# Paleta fija de 20 colores para CIFAR-FS
PALETTE_20 = px.colors.qualitative.Dark24[:20]


def run_tsne_and_save_fix_values(X, y, name, TSNE_DIR, perplexity=30, lr=200):
    """
    Normalizar evita que unas pocas dimensiones con mayor escala dominen la proyección, lo que puede crear separaciones artificiales
    Al estandarizar las características, la visualización refleja mejor la estructura real 
    del espacio de embeddings y hace las comparaciones entre métodos más justas
    """
    out_path = os.path.join(TSNE_DIR, f"{name}.html")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True) 

    print(f"Running t-SNE for {name} with {X.shape[0]} samples...")

    X = StandardScaler().fit_transform(X) 

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=lr,
        init='pca',
        random_state=0
    )

    X_2d = tsne.fit_transform(X)

    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=y,
        color_discrete_sequence=PALETTE_20,
        title=f"{name} (perplexity={perplexity}, n={X.shape[0]})"
    )

    out_path = os.path.join(TSNE_DIR, f"{name}.html")
    fig.write_html(out_path)
    print(f"Saved: {out_path}\n")


def run_tsne_and_save_adaptable(X, y, name, TSNE_DIR):
    X = StandardScaler().fit_transform(X)

    perplexity = min(30, X.shape[0] - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=0
    )

    X_2d = tsne.fit_transform(X)

    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=y,
        color_discrete_sequence=PALETTE_20,
        title=f"{name} (perplexity={perplexity}, n={X.shape[0]})"
    )

    out_path = os.path.join(TSNE_DIR, f"{name}.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)



def run_tsne_and_save_text(X, y, name, TSNE_DIR):
    """
    Parametros especiales para texto
    """
    X = StandardScaler().fit_transform(X) 

    out_path = os.path.join(TSNE_DIR, f"{name}.html")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True) 

    print("Caso especial texto")
    perplexity = 3

    tsne = TSNE(
        n_components=2,
        perplexity=3,
        learning_rate="auto",
        random_state=0
    )

    X_2d = tsne.fit_transform(X)

    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=y,
        color_discrete_sequence=PALETTE_20,
        title=f"{name} (perplexity={perplexity}, n={X.shape[0]})"
    )

    out_path = os.path.join(TSNE_DIR, f"{name}.html")
    fig.write_html(out_path)
    print(f"Saved: {out_path}\n")



