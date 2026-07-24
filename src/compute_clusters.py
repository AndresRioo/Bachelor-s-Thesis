import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from model.res12 import Res12
from model.swin_transformer import swin_tiny

from utils import (
    transform_val,
    transform_val_224,
    transform_val_cifar,
    transform_val_224_cifar
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model(args, device):
    """
    Construye el backbone visual y carga sus pesos preentrenados.

    Args:
        args:
            Argumentos de línea de comandos. Debe contener
            'backbone' y 'dataset'.

        device (torch.device):
            Dispositivo donde se cargará el modelo.

    Returns:
        torch.nn.Module:
            Modelo en modo evaluación con los pesos cargados.
    """
    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)

        checkpoint_path = f'./checkpoints/ResNet-{args.dataset}.pth'

        checkpoint = torch.load(checkpoint_path, weights_only=False)['params']

        checkpoint = {k[8:]: v for k, v in checkpoint.items()}

    elif args.backbone == 'swin':

        model = swin_tiny().to(device)

        checkpoint_path = f'./checkpoints/Swin-Tiny-{args.dataset}.pth'

        checkpoint = torch.load(checkpoint_path, weights_only=False)['params']

    else:
        raise ValueError("Unsupported backbone")

    model_dict = model.state_dict()

    checkpoint = {
        k: v for k, v in checkpoint.items()
        if k in model_dict
    }

    model.load_state_dict(checkpoint)

    model.eval()

    return model


def build_dataset(args):
    """
    Construye el conjunto de entrenamiento utilizado para extraer
    las representaciones visuales de las clases.

    Args:
        args:
            Argumentos de configuración. Debe contener
            'dataset' y 'backbone'.

    Returns:
        torchvision.datasets.ImageFolder:
            Dataset con las transformaciones apropiadas para el
            backbone seleccionado.
    """

    if args.dataset == 'MiniImageNet':

        path = 'MINIIMAGENET/miniimagenet/train'

        transform = transform_val if args.backbone == 'resnet' else transform_val_224

    elif args.dataset == 'CIFAR-FS':

        path = 'CIFAR-FS/cifar-fs/train'

        transform = transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar

    elif args.dataset == 'FC100':

        path = '/path/to/your/fc100/train'

        transform = transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar

    else:
        raise ValueError("Unsupported dataset")

    dataset = ImageFolder(path, transform=transform)

    return dataset


@torch.no_grad()
def extract_class_features(model, loader, idx_to_class, device):
    """
    Extrae una representación media normalizada para cada clase.

    Args:
        model (torch.nn.Module):
            Backbone utilizado para obtener embeddings visuales.

        loader (DataLoader):
            Loader del conjunto de entrenamiento.

        idx_to_class (dict):
            Mapeo índice -> nombre de clase.

        device (torch.device):
            Dispositivo de inferencia.

    Returns:
        dict[str, torch.Tensor]:
            Embedding medio normalizado de cada clase.
    """

    class_features = {}

    for images, labels in loader:

        images = images.to(device)

        feats = model(images)
        feats = F.normalize(feats, dim=1)

        feats = feats.cpu()

        # Agrupa los embeddings por clase
        for feat, label in zip(feats, labels):

            class_name = idx_to_class[label.item()]

            if class_name not in class_features:
                class_features[class_name] = []

            class_features[class_name].append(feat)

    class_means = {}

    # Calcula el prototipo visual de cada clase
    for class_name, feats in class_features.items():

        feats = torch.stack(feats)

        mean_feat = feats.mean(dim=0)
        mean_feat = F.normalize(mean_feat, dim=0)

        class_means[class_name] = mean_feat

    return class_means


def load_semantic_features(args):
    """
    Carga las representaciones semánticas precomputadas de cada
    clase y las normaliza.

    Args:
        args:
            Argumentos de configuración. Debe contener
            'dataset', 'mode' y 'text_type'.

    Returns:
        dict[str, torch.Tensor]:
            Embeddings semánticos normalizados para cada clase.
    """

    if 'ImageNet' in args.dataset:

        semantic_path = f'./semantic/imagenet_semantic_{args.mode}_{args.text_type}.pth'

    else:

        semantic_path = f'./semantic/cifar100_semantic_{args.mode}_{args.text_type}.pth'

    semantic = torch.load(semantic_path, weights_only=False)['semantic_feature']

    semantic = {
        k: F.normalize(v.float(), dim=0)
        for k, v in semantic.items()
    }

    return semantic


def build_feature_matrix(class_names, features_dict):
    """
    Convierte un diccionario de embeddings en una matriz ordenada
    según la lista de clases proporcionada.

    Args:
        class_names (list[str]):
            Orden de clases deseado.

        features_dict (dict):
            Diccionario clase -> embedding.

    Returns:
        np.ndarray:
            Matriz de tamaño [num_classes, feature_dim].
    """

    matrix = []

    for cls in class_names:
        matrix.append(features_dict[cls].numpy())

    matrix = np.stack(matrix)

    return matrix


def compute_hierarchy(feature_matrix):
    """
    Construye un clustering jerárquico aglomerativo a partir de una
    matriz de características.

    Args:
        feature_matrix (np.ndarray):
            Matriz [num_classes, feature_dim] con los embeddings
            de cada clase.

    Returns:
        np.ndarray:
            Linkage matrix producida por scipy, utilizada para
            generar dendrogramas y extraer clusters.
    """

    # Distancias coseno entre todas las parejas de clases
    distances = pdist(feature_matrix, metric='cosine')

    # Clustering jerárquico mediante average linkage
    Z = linkage(distances, method='average')

    return Z


def save_dendrogram(Z, class_names, output_path):
    """
    Genera y guarda una visualización del árbol jerárquico.

    Args:
        Z (np.ndarray):
            Linkage matrix generada por scipy.

        class_names (list[str]):
            Etiquetas utilizadas en las hojas del dendrograma.

        output_path (str):
            Ruta donde se guardará la figura.

    Returns:
        None
    """
    plt.figure(figsize=(18, 8))

    dendrogram(
        Z,
        labels=class_names,
        leaf_rotation=90
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def compute_cluster_centers(
    Z,
    class_names,
    visual_class_means,
    levels
):
    """
    Calcula la composición, centro y radio de los clusters para
    distintos niveles del árbol jerárquico.

    Args:
        Z (np.ndarray):
            Linkage matrix del clustering jerárquico.

        class_names (list[str]):
            Lista ordenada de nombres de clase.

        visual_class_means (dict):
            Prototipos visuales de cada clase.

        levels (list[int]):
            Número de clusters deseado para cada nivel.

    Returns:
        dict:
            Información de cada nivel de clustering que incluye:

            - class_to_cluster
            - cluster_to_classes
            - cluster_centers
            - cluster_radii
    """

    results = {}

    for n_clusters in levels:

        cluster_ids = fcluster(
            Z,
            t=n_clusters,
            criterion='maxclust'
        )

        class_to_cluster = {}

        cluster_to_classes = {}

        # Radio máximo de cada cluster respecto a su centro
        cluster_radii = {}

        # Asigna cada clase al cluster correspondiente
        for cls, cluster_id in zip(class_names, cluster_ids):

            class_to_cluster[cls] = int(cluster_id)

            if int(cluster_id) not in cluster_to_classes:
                cluster_to_classes[int(cluster_id)] = []

            cluster_to_classes[int(cluster_id)].append(cls)

        MIN_CLUSTER_SIZE = 2

        # Elimina clusters degenerados con menos de dos clases
        cluster_to_classes = {
            cid: classes
            for cid, classes in cluster_to_classes.items()
            if len(classes) >= MIN_CLUSTER_SIZE
        }

        # Reconstruye el mapeo usando únicamente los clusters válidos
        class_to_cluster = {}

        for cid, classes in cluster_to_classes.items():

            for cls in classes:

                class_to_cluster[cls] = cid

        cluster_centers = {}

        # Calcula el centroide visual de cada cluster
        for cluster_id, cluster_classes in cluster_to_classes.items():

            feats = []

            for cls in cluster_classes:
                feats.append(visual_class_means[cls])

            feats = torch.stack(feats)

            center = feats.mean(dim=0)

            center = F.normalize(center, dim=0)

            cluster_centers[int(cluster_id)] = center
        
            distances = []

            for cls in cluster_classes:

                feat = visual_class_means[cls]

                dist = 1 - F.cosine_similarity(
                    feat.unsqueeze(0),
                    center.unsqueeze(0)
                ).item()

                distances.append(dist)
            
            radius = max(distances)
            cluster_radii[int(cluster_id)] = radius

        results[n_clusters] = {
            'class_to_cluster': class_to_cluster,
            'cluster_to_classes': cluster_to_classes,
            'cluster_centers': cluster_centers,
            'cluster_radii': cluster_radii
        }

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR-FS',
        choices=['MiniImageNet', 'CIFAR-FS', 'FC100']
    )

    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet',
        choices=['resnet', 'swin']
    )

    parser.add_argument(
        '--cluster-mode',
        type=str,
        default='visual',
        choices=['visual', 'semantic']
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='clip',
        choices=['clip', 'bert']
    )

    parser.add_argument(
        '--text_type',
        type=str,
        default='gpt',
        choices=['gpt', 'name', 'definition']
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=8
    )

    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=[32, 16, 8]
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./precomputed_clusters'
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    dataset = build_dataset(args)

    idx_to_class = dataset.class_to_idx
    idx_to_class = {v: k for k, v in idx_to_class.items()}

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("Loading backbone...")
    model = build_model(args, device)

    print("Extracting visual class means...")
    visual_class_means = extract_class_features(
        model,
        loader,
        idx_to_class,
        device
    )

    class_names = sorted(list(visual_class_means.keys()))

    print(f"Building {args.cluster_mode} clustering features...")

    if args.cluster_mode == 'visual':

        clustering_features = visual_class_means

    elif args.cluster_mode == 'semantic':

        semantic_features = load_semantic_features(args)

        clustering_features = {
            cls: semantic_features[cls]
            for cls in class_names
        }

    else:
        raise ValueError("Invalid clustering mode")

    feature_matrix = build_feature_matrix(
        class_names,
        clustering_features
    )

    print("Computing hierarchical clustering...")
    Z = compute_hierarchy(feature_matrix)

    dendrogram_path = os.path.join(
        args.output_dir,
        f'dendrogram_{args.dataset}_{args.cluster_mode}.png'
    )

    print("Saving dendrogram...")
    save_dendrogram(
        Z,
        class_names,
        dendrogram_path
    )

    print("Computing cluster centers...")
    levels_data = compute_cluster_centers(
        Z,
        class_names,
        visual_class_means,
        args.levels
    )

    # CLUSTER DEBUG ANALYSIS
    DEBUG_DIR = os.path.join(
        args.output_dir,
        f"cluster_debug_{args.cluster_mode}"
    )
    os.makedirs(DEBUG_DIR, exist_ok=True)

    for level in [32, 16, 8]:

        print()
        print("=" * 80)
        print(f"LEVEL {level}")
        print("=" * 80)

        level_info = levels_data[level]

        cluster_to_classes = level_info["cluster_to_classes"]
        cluster_radii = level_info["cluster_radii"]
        cluster_centers = level_info["cluster_centers"]

        txt_path = os.path.join(
            DEBUG_DIR,
            f"clusters_level_{level}.txt"
        )

        with open(txt_path, "w") as f:
            
            f.write("=" * 80 + "\n")
            f.write(f"LEVEL {level}\n")
            f.write("=" * 80 + "\n\n")

            sizes = np.array([
                len(classes)
                for classes in cluster_to_classes.values()
            ])

            f.write("Cluster sizes\n")
            f.write(f"Num clusters : {len(sizes)}\n")
            f.write(f"Mean size    : {sizes.mean():.2f}\n")
            f.write(f"Min size     : {sizes.min()}\n")
            f.write(f"Max size     : {sizes.max()}\n")
            f.write(f"Sorted sizes : {np.sort(sizes)}\n\n")

            for cluster_id, classes in sorted(
                cluster_to_classes.items(),
                key=lambda x: len(x[1]),
                reverse=True
            ):

                f.write(
                    f"Cluster {cluster_id} | "
                    f"size={len(classes)} | "
                    f"radius={cluster_radii[cluster_id]:.4f}\n"
                )

                f.write(
                    ", ".join(classes)
                )

                f.write("\n\n")

        # --------------------------------------------------------
        # CLUSTER SIZE ANALYSIS
        # --------------------------------------------------------

        sizes = []

        for cluster_id, classes in cluster_to_classes.items():
            sizes.append(len(classes))

        sizes = np.array(sizes)

        print("\nCluster sizes")
        print(f"Num clusters      : {len(sizes)}")
        print(f"Mean size         : {sizes.mean():.2f}")
        print(f"Min size          : {sizes.min()}")
        print(f"Max size          : {sizes.max()}")

        print("\nSorted sizes")
        print(np.sort(sizes))

        print("\nCLUSTER CONTENTS")
        print("-" * 80)

        for cluster_id, classes in sorted(
            cluster_to_classes.items(),
            key=lambda x: len(x[1]),
            reverse=True
        ):

            print(
                f"\nCluster {cluster_id}"
                f" | size={len(classes)}"
                f" | radius={cluster_radii[cluster_id]:.4f}"
            )

            print(classes)

    # RADIUS HISTOGRAM ANALYSIS
    print("Generating radius histograms...")

    HIST_DIR = os.path.join(
        args.output_dir,
        f"radius_analysis_{args.cluster_mode}"
    )
    os.makedirs(HIST_DIR, exist_ok=True)

    for level in args.levels:

        cluster_radii_dict = levels_data[level]['cluster_radii']

        radii = np.array(list(cluster_radii_dict.values()))

        # HISTOGRAMA
        plt.figure(figsize=(8, 5))

        plt.hist(
            radii,
            bins=20,
            edgecolor='black'
        )

        plt.title(f"Cluster Radius Distribution - Level {level}")
        plt.xlabel("Cluster Radius")
        plt.ylabel("Frequency")

        plt.tight_layout()

        hist_path = os.path.join(
            HIST_DIR,
            f"radius_hist_level_{level}.png"
        )

        plt.savefig(hist_path)
        plt.close()

        print(f"Saved histogram for level {level}")


    save_path = os.path.join(
        args.output_dir,
        f'clusters_{args.dataset}_{args.backbone}_{args.cluster_mode}.pth'
    )

    print("Saving results...")

    torch.save({
        'dataset': args.dataset,
        'backbone': args.backbone,
        'cluster_mode': args.cluster_mode,
        'levels': levels_data,
        'class_means': visual_class_means,
        'linkage_matrix': Z,
        'class_names': class_names
    }, save_path)

    print()
    print("Finished.")
    print(f"Saved file: {save_path}")
    print(f"Dendrogram: {dendrogram_path}")


# Hierarchical clustering semana 24 : 10 de mayo

# este codigo lo que hace es generar las clases que forman el cluster a diferentes niveles
# estos clusters se pueden calcular de 2 maneras 

# vision : media de todas las imagenes de train 
# semantica : semejanza semantica entre las clases

# FORMULA DEL TRAIN (USANDO RADIO FIJO) PROPUESTA 2
# L=Lrecon​+λmax(0,m−dcos​(ck​,py​))

# FORMULA CON MARGEN DEL CLUSTER PROPUESTA FINAL
# L=Lrecon​+λmax(0,(rk​*(1+m)−d(ck​,py​))

# si distance < danger_radius entonces penaliza 

# EXPLICACION PARAMETROS 

# CLUSTER LEVELS : n clusters totales 
# MARGIN : tamaño del radio
# LAMBDA : Peso dado a la loss


