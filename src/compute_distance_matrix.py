import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform


# Cargar representaciones sen
semantic = torch.load(
    './semantic/cifar100_semantic_clip_gpt.pth',
    weights_only=False
)['semantic_feature']

# Convierte cada tensor de PyTorch a NumPy
semantic = {
    k: v.numpy()
    for k, v in semantic.items()
}

# Agrupa todos los vectores en una única matriz
features = np.stack(
    [semantic[c] for c in semantic.keys()]
)
 
# Calcula la matriz completa de distancias coseno entre clases
semantic_dist = squareform(
    pdist(features, metric='cosine')
)

np.save(
    'semantic_tree_dist.npy',
    semantic_dist
)

print("Semantic distance matrix saved as 'semantic_tree_dist.npy' !")