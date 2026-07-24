import os
import sys
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wandb


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from torchvision.datasets import ImageFolder
from sklearn.preprocessing import normalize

# from data.tiered_imagenet import tieredImageNet
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.samplers import CategoriesSampler
from logger import loggers
from model.res12 import Res12
from model.swin_transformer import swin_tiny
from utils import set_seed, Cosine_classifier, count_95acc, count_kacc, transform_val_cifar, transform_val, transform_val_224, transform_val_224_cifar

from sklearn.manifold import TSNE
import plotly.express as px
import os

from sklearn.preprocessing import StandardScaler

from method_original._tsne import run_tsne_and_save_adaptable, run_tsne_and_save_text, run_tsne_and_save_fix_values
from method_original._confusion_matrix import save_confusion_matrix, compute_confusion_matrices

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

import method_original as method

from analysis.tsne_utils import (
    generate_tsne_plot
)

from analysis.confusion_utils import (
    save_interactive_cm,
    compute_metrics,
    compute_ova,
    top_confusions,
    per_class_accuracy
)

from analysis.clustering_utils import (
    save_dendrogram,
    build_mutual_confusion_distance
)

from analysis.common_utils import safe_pca


def safe_pca(X, max_components=50):
    n_components = min(max_components, X.shape[0], X.shape[1])
    return PCA(n_components=n_components).fit_transform(X)



if __name__ == '__main__':

    sys.modules['method'] = method

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'mean'])
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--feat-size', type=int, default=640)
    parser.add_argument('--semantic-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'bert'])
    parser.add_argument('--text_type', type=str, default='gpt',
                        choices=['gpt', 'name', 'definition'])
    parser.add_argument('--dataset', type=str, default='TieredImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'swin'])
    
    parser.add_argument('--run-tsne', type=int, default=1, help='1: calcular t-SNE, 0: saltarlo')
    parser.add_argument('--tsne-text', type=int, default=1, help='1: calcular t-SNE, 0: saltarlo')
    parser.add_argument('--tsne-vision', type=int, default=1, help='1: calcular t-SNE, 0: saltarlo')
    parser.add_argument('--tsne-sem', type=int, default=1, help='1: calcular t-SNE, 0: saltarlo')
    parser.add_argument('--tsne-prototype', type=int, default=1, help='1: calcular t-SNE, 0: saltarlo')

    parser.add_argument('--run-confusion-matrix', type=int, default=1, help='1: calcular, 0: saltarlo')
    parser.add_argument('--run-hierarchical', type=int, default=1)
    parser.add_argument('--run-metrics', type=int, default=1)
    parser.add_argument('--analysis-dir', type=str, default='results_hierarchical_margin')

    # ARGUMENTOS NUEVOS PARA EL CLUSTERING
    parser.add_argument(
        '--cluster-level',
        type=int,
        default=16
    )

    parser.add_argument(
        '--radius-scale',
        type=float,
        default=0.05
    )

    parser.add_argument(
        '--cluster-lambda',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--cluster-mode',
        type=str,
        default='visual',
        choices=['visual', 'semantic']
    )

    args = parser.parse_args()


    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)


    args.work_dir = '{}_{}_{}_{}_{}_{}_cluster{}_radiusscale{}_lambda{}_mode{}'.format(
        args.backbone,
        args.dataset,
        args.mode,
        args.text_type,
        args.center,
        args.shot,
        args.cluster_level,
        args.radius_scale,
        args.cluster_lambda,
        args.cluster_mode
    )



    log = loggers('test_{}'.format(args.dataset))
    log.info(vars(args))
    set_seed(args.seed)

    if args.dataset == 'TieredImageNet':
        args.num_workers = 0

    if args.dataset == 'MiniImageNet':
        args.test = 'MINIIMAGENET/miniimagenet/test'
        test_dataset = ImageFolder(args.test, transform=transform_val if args.backbone == 'resnet' else transform_val_224)
    elif args.dataset == 'FC100':
        args.test = '/path/to/your/fc100/test'
        test_dataset = ImageFolder(args.test, transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        args.test = 'CIFAR-FS/cifar-fs/test'
        test_dataset = ImageFolder(args.test, transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)
    elif args.dataset == 'TieredImageNet':
        test_dataset = tieredImageNet(setname='test')

        if args.backbone == 'resnet':
            args.test = '/path/to/your/tiredimagenet/test'
            test_dataset = ImageFolder(args.test, transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    idx_to_class = test_dataset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}

    val_sampler = CategoriesSampler(test_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=test_dataset, batch_sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path, weights_only=False)['params']
        checkpoint = {k[8:]: v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path, weights_only=False)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        
    print(len(checkpoint))
    model.load_state_dict(checkpoint)
    model.eval()

    Model_PATH = os.path.join(args.work_dir, 'epoch_best.pth')
    H = torch.load(Model_PATH, weights_only=False)

    #fusion = H['G']  # original
    fusion = H['G'].to(device).eval()
    
    best_epoch = H['epoch']
    best_acc = H['acc']
    best_k = H['k']
    log.info('best epoch: %d %2f' % (best_epoch, best_acc * 100))
    log.info('best k: %2f' % (float(best_k)))

    if 'ImageNet' in args.dataset:
        semantic = torch.load('./semantic/imagenet_semantic_{}_{}.pth'.format(args.mode, args.text_type), weights_only=False)['semantic_feature']
    else:
        semantic = torch.load('./semantic/cifar100_semantic_{}_{}.pth'.format(args.mode, args.text_type), weights_only=False)['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}



    ks = np.arange(0, 101) * 0.01
    label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)
    with torch.no_grad():
        A_acc = []
        P_acc = []
        G_acc = []
        for data, labels in tqdm(val_loader):
            data = data.to(device)
            data = model(data).view(data.size(0), -1)  

            # T-SNE VISION ENCODER
            # vision_emb = model(data).view(data.size(0), -1)


            n_support = args.shot * args.test_way
            support, query = data[:n_support], data[n_support:]

            proto = support.reshape(args.shot, args.test_way, -1).mean(dim=0)
            s = torch.stack([semantic[idx_to_class[l.item()]] for l in labels[:n_support]]).to(device)

            # T-SNE TEXT ENCODER
            # text_emb = semantic[idx_to_class[label]]


            gen_proto = fusion(s, support)

            # T-SNE SEMALIGN
            # semalign_emb = fusion(text_emb, vision_emb)


            gen_proto = gen_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            dist0, predict0 = Cosine_classifier(proto, query)
            dist1, predict1 = Cosine_classifier(gen_proto, query)
            P_acc.append(((predict0 == label).sum() / len(label)).item())
            G_acc.append(((predict1 == label).sum() / len(label)).item())

            A_acc.append(count_kacc(proto, gen_proto, query, torch.tensor(float(best_k)), args))

        P_acc, P_95 = count_95acc(np.array(P_acc))
        G_acc, G_95 = count_95acc(np.array(G_acc))
        A_acc_mean, A_acc_95 = count_95acc(np.array(A_acc))

        import wandb
        from datetime import datetime

        today = datetime.now().strftime("%d%b").lower()

        from dotenv import load_dotenv
        import os
        from pathlib import Path

        ROOT_DIR = Path(__file__).resolve().parent.parent

        load_dotenv(ROOT_DIR / ".env")

        os.environ["WANDB_SILENT"] = "true"

        WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        EXPERIMENT_TAG = os.getenv("EXPERIMENT_TAG")

        wandb.login(key=WANDB_API_KEY)

        wandb.init(
            project="semfew-tfg-test",
            name=f"hc_margen_{today}_{args.work_dir}",
            tags=[
                args.dataset,
                args.backbone,
                f"{args.shot}-shot",
                args.mode,
                "hc_margen",
                today,
                f"cluster-{args.cluster_level}",
                f"radius_scale-{args.radius_scale}",
                f"cluster_lambda-{args.cluster_lambda}",
                f"cluster_mode-{args.cluster_mode}",
                EXPERIMENT_TAG
            ],
            config=vars(args),
            settings=wandb.Settings(
                init_timeout=300,
                start_method="thread"
            )
        )

        wandb.run.summary["final_mix_acc"] = A_acc_mean * 100
        wandb.run.summary["final_mix_acc_95"] = A_acc_95 * 100

        wandb.run.summary["final_proto_acc"] = P_acc * 100
        wandb.run.summary["final_proto_acc_95"] = P_95 * 100

        wandb.run.summary["final_gen_acc"] = G_acc * 100
        wandb.run.summary["final_gen_acc_95"] = G_95 * 100

        wandb.run.summary["final_gap"] = (
            A_acc_mean * 100 - P_acc * 100
        )

        wandb.run.summary["final_mix_acc_str"] = (
            f"{A_acc_mean * 100:.2f} ± {A_acc_95 * 100:.2f}"
        )

        wandb.run.summary["best_k"] = float(best_k)

        wandb.run.summary["cluster_level"] = args.cluster_level
        wandb.run.summary["radius_scale"] = args.radius_scale
        wandb.run.summary["cluster_lambda"] = args.cluster_lambda
        wandb.run.summary["cluster_mode"] = args.cluster_mode

        wandb.finish()
        
        # print antiguo
        log.info('max |k: %16s |mix acc: %.2f+%.2f%% |gap: %.2f' % (
            best_k, A_acc_mean * 100, A_acc_95 * 100, A_acc_mean * 100 - P_acc * 100))
        log.info('ACC:|proto acc: %.2f+%.2f%% |gen acc: %.2f+%.2f%%' % (
            P_acc * 100, P_95 * 100, G_acc * 100, G_95 * 100))
        
        # print nuevo
        log.info("")
        log.info("========== TEST RESULTS ==========")
        log.info(f"Dataset: {args.dataset} | Backbone: {args.backbone}")
        log.info(f"Setting: {args.test_way}-way {args.shot}-shot | Query: {args.query}")
        log.info("----------------------------------")
        log.info(f"Best k        : {float(best_k):.3f}")
        log.info(f"Mix Accuracy  : {A_acc_mean * 100:.2f} ± {A_acc_95 * 100:.2f} %")
        log.info(f"Proto Accuracy: {P_acc * 100:.2f} ± {P_95 * 100:.2f} %")
        log.info(f"Gen Accuracy  : {G_acc * 100:.2f} ± {G_95 * 100:.2f} %")
        log.info(f"Gap (Mix-Proto): {(A_acc_mean * 100 - P_acc * 100):.2f} %")
        log.info("==================================")


    # ==========================================================
    # EXPERIMENT RESULTS
    # ==========================================================
    #
    # Save the final evaluation results together wi
    #
    # Generated file:
    #   results.txt
    #
    # ==========================================================

    ANALYSIS_DIR = os.path.join(
        BASE_DIR,
        args.analysis_dir,
        args.work_dir
    )
    
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Quick summary
    results_path = os.path.join(ANALYSIS_DIR, "results.txt")

    # for data analysis 
    results_excel_path = os.path.join(
        ANALYSIS_DIR,
        "results.xlsx"
    )

    with open(results_path, "w") as f:

        f.write("==================================================\n")
        f.write("HIERARCHICAL-MARGIN EVALUATION RESULTS\n")
        f.write("==================================================\n\n")

        f.write("Experiment Configuration\n")
        f.write("------------------------\n")
        f.write(f"Dataset          : {args.dataset}\n")
        f.write(f"Backbone         : {args.backbone}\n")
        f.write(f"Few-Shot Setting : {args.test_way}-way {args.shot}-shot\n")
        f.write(f"Query Samples    : {args.query}\n")
        f.write(f"Semantic Mode    : {args.mode}\n")
        f.write(f"Text Type        : {args.text_type}\n")
        f.write(f"Center Strategy  : {args.center}\n\n")

        f.write("Hierarchical-Margin Parameters\n")
        f.write("------------------------\n")
        f.write(f"Cluster Level    : {args.cluster_level}\n")
        f.write(f"Radius Scale     : {args.radius_scale}\n")
        f.write(f"Cluster Lambda   : {args.cluster_lambda}\n")
        f.write(f"Cluster Mode     : {args.cluster_mode}\n\n")

        f.write("Model Performance\n")
        f.write("------------------------\n")
        f.write(f"Best Fusion Weight (k) : {float(best_k):.3f}\n")

        f.write(
            f"Mix Accuracy           : "
            f"{A_acc_mean * 100:.2f} ± {A_acc_95 * 100:.2f} %\n"
        )

        f.write(
            f"Visual Prototype       : "
            f"{P_acc * 100:.2f} ± {P_95 * 100:.2f} %\n"
        )

        f.write(
            f"SemAlign Prototype     : "
            f"{G_acc * 100:.2f} ± {G_95 * 100:.2f} %\n"
        )

        f.write(
            f"Gain (Mix - Visual)    : "
            f"{(A_acc_mean - P_acc) * 100:.2f} %\n"
        )

        f.write(
            f"Gain (SemAlign - Visual): "
            f"{(G_acc - P_acc) * 100:.2f} %\n"
        )

        f.write("\n==================================================\n")
        

    df_results = pd.DataFrame([
        {
            "dataset": args.dataset,
            "backbone": args.backbone,

            "mode": args.mode,
            "text_type": args.text_type,
            "center": args.center,

            "way": args.test_way,
            "shot": args.shot,
            "query": args.query,

            "seed": args.seed,

            "cluster_level": args.cluster_level,
            "radius_scale": args.radius_scale,
            "cluster_lambda": args.cluster_lambda,
            "cluster_mode": args.cluster_mode,

            "best_k": float(best_k),

            "mix_accuracy": A_acc_mean,
            "mix_ci95": A_acc_95,

            "proto_accuracy": P_acc,
            "proto_ci95": P_95,

            "gen_accuracy": G_acc,
            "gen_ci95": G_95,

            "gain_mix_vs_proto":
                A_acc_mean - P_acc,

            "gain_gen_vs_proto":
                G_acc - P_acc
        }
    ])



    with pd.ExcelWriter(
        results_excel_path
    ) as writer:

        df_results.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

    log.info(f"Resultados guardados en: {results_path}")
    log.info("")



# ==========================================================
# TSNE ANALYSIS
# ==========================================================
#
# Generates qualitative visualizations of the embedding space.
#
# Generated visualizations:
#
# 1. Embeddings
#    Global representation of the embedding space.
#
#    - Text Encoder:
#        Semantic CLIP embeddings.
#
#    - Vision Encoder:
#        Visual backbone embeddings.
#
#    - SemAlign:
#        Embeddings after semantic fusion.
#
#    - Vision vs SemAlign:
#        Direct comparison between visual and fused spaces.
#
# 2. Prototype Visualization
#    Single-episode analysis showing:
#
#    - Support samples
#    - Query samples
#    - Visual prototypes
#    - Reconstructed (SemAlign) prototypes
#
#    Purpose:
#    Analyze how semantic fusion modifies prototype locations
#    within a specific few-shot episode.
#
# 3. Prototype Distribution
#    Multi-episode analysis of prototype variability.
#
#    Purpose:
#    Evaluate the consistency of visual and SemAlign
#    prototypes across different episodes.
#
# 4. Query Embeddings
#    Direct comparison between visual query embeddings
#    and SemAlign query embeddings.
#
#    Purpose:
#    Analyze how semantic fusion modifies query
#    representations.
#
# Overall Purpose:
#    Understand how semantic fusion modifies the geometry
#    of the embedding space at different levels:
#
#    - Global feature space
#    - Prototype representations
#    - Prototype stability across episodes
#    - Query embeddings
#
# ==========================================================
    if args.run_tsne == 1:

        TSNE_DIR = os.path.join(ANALYSIS_DIR, "tsne")

        EMBEDDINGS_DIR = os.path.join(TSNE_DIR, "embeddings")
        PROTO_VIS_DIR = os.path.join(TSNE_DIR, "prototype_visualization")
        PROTO_DIST_DIR = os.path.join(TSNE_DIR, "prototype_distribution")
        QUERY_DIR = os.path.join(TSNE_DIR, "query_embeddings")

        for d in [
            TSNE_DIR,
            EMBEDDINGS_DIR,
            PROTO_VIS_DIR,
            PROTO_DIST_DIR,
            QUERY_DIR
        ]:
            os.makedirs(d, exist_ok=True)

        vision_feats = []
        vision_labels = []

        semalign_feats = []
        semalign_labels = []

        # Limit the number of episodes to keep the visualization readable
        MAX_EPISODES = 10
        episode_count = 0

        # ===== TEXT ENCODER: 1 embedding per class  =====
        text_feats = []
        text_labels = []

        for cls_id, cls_name in idx_to_class.items():
            text_feats.append(semantic[cls_name].cpu().numpy())
            text_labels.append(cls_name)

        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Collecting embeddings for t-SNE"):
                data = data.to(device)
                feats = model(data).view(data.size(0), -1)

                n_support = args.shot * args.test_way
                support, query = feats[:n_support], feats[n_support:]
                support_labels = labels[:n_support]
                query_labels = labels[n_support:]

                # Vision encoder (usamos solo queries para que no se vea el soporte repetido)
                vision_feats.append(query.cpu().numpy())
                vision_labels.extend([idx_to_class[l.item()] for l in query_labels])

                # Semantic features associated with episode classes
                episode_classes = support_labels.view(args.shot, args.test_way)[0]

                # ============================
                # SEMALIGN GLOBAL
                # ============================
                class_to_sem = {
                    idx_to_class[c.item()]: semantic[idx_to_class[c.item()]]
                    for c in episode_classes
                }

                text_query = torch.stack([
                    class_to_sem[idx_to_class[l.item()]] for l in query_labels
                ]).to(device)

                # Generate one fused embedding per query sample
                semalign_query = fusion(text_query, query)

                # Store embeddings for later visualization
                semalign_feats.append(semalign_query.cpu().numpy())
                semalign_labels.extend([idx_to_class[l.item()] for l in query_labels])

                episode_count += 1
                if episode_count >= MAX_EPISODES:
                    break

        vision_feats = np.concatenate(vision_feats, axis=0)
        semalign_feats = np.concatenate(semalign_feats, axis=0)
        text_feats = np.stack(text_feats, axis=0)


        # VISION VS SEMALIGN COMPARISON

        combined_feats = np.concatenate([vision_feats, semalign_feats])
        combined_labels = vision_labels + semalign_labels

        combined_types = (
            ["vision"] * len(vision_feats)
            + ["semalign"] * len(semalign_feats)
        )

        TSNE_PERPLEXITY = 30
        TSNE_LR = 200

        generate_tsne_plot(
            combined_feats,
            combined_labels,
            os.path.join(
                EMBEDDINGS_DIR,
                "vision_vs_semalign.html"
            ),
            title="Vision vs SemAlign",
            symbols=combined_types
        )

        generate_tsne_plot(
            semalign_feats,
            semalign_labels,
            os.path.join(
                EMBEDDINGS_DIR,
                "semalign_encoder.html"
            ),
            title="SemAlign Embeddings",
            perplexity=20,
            learning_rate=150
        )        

        generate_tsne_plot(
            text_feats,
            text_labels,
            os.path.join(
                EMBEDDINGS_DIR,
                "text_encoder.html"
            ),
            title="Text Encoder Embeddings",
            perplexity=10,
            learning_rate=100
        )        
        
        generate_tsne_plot(
            vision_feats,
            vision_labels,
            os.path.join(
                EMBEDDINGS_DIR,
                "vision_encoder.html"
            ),
            title="Vision Encoder Embeddings",
            perplexity=40,
            learning_rate=300
        )

        # ============================
        # PROTOTYPE VISUALIZATION (Paper-style Figure 5)
        # ============================

        print("\n==== PROTOTYPE VISUALIZATION (Support / Query / Proto / SemAlign) ====\n")

        support_feats = []
        support_labels = []

        query_feats = []
        query_labels = []

        proto_feats = []
        proto_labels = []

        genproto_feats = []
        genproto_labels = []

        # Use a single episode to keep the plot interpretable
        for data, labels in val_loader:

            data = data.to(device)
            feats = model(data).view(data.size(0), -1)

            n_support = args.shot * args.test_way
            support, query = feats[:n_support], feats[n_support:]

            support_labels_episode = labels[:n_support]
            query_labels_episode = labels[n_support:]

            # ---------------------------
            # REAL PROTOTYPE
            # ---------------------------

            proto = support.reshape(args.shot, args.test_way, -1).mean(dim=0)

            # ---------------------------
            # SEMALIGN PROTOTYPE
            # ---------------------------

            # Semantic embedding associated with each episode class
            episode_classes = support_labels_episode.view(args.shot, args.test_way)[0]

            s_class = torch.stack([
                semantic[idx_to_class[c.item()]] for c in episode_classes
            ]).to(device)

            # Repeat semantic embedding for every support sample
            s_expanded = s_class.repeat(args.shot, 1)

            gen_proto = fusion(s_expanded, support)
            gen_proto = gen_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            # ---------------------------
            # STORE FEATURES
            # ---------------------------

            support_feats.append(support.detach().cpu().numpy())
            query_feats.append(query.detach().cpu().numpy())

            proto_feats.append(proto.detach().cpu().numpy())
            genproto_feats.append(gen_proto.detach().cpu().numpy())

            support_labels.extend([idx_to_class[l.item()] for l in support_labels_episode])
            query_labels.extend([idx_to_class[l.item()] for l in query_labels_episode])

            proto_labels.extend([idx_to_class[l.item()] for l in support_labels_episode[:args.test_way]])
            genproto_labels.extend([idx_to_class[l.item()] for l in support_labels_episode[:args.test_way]])

            break  # Only one episode is used to keep the visualization readable

        support_feats = np.concatenate(support_feats)
        query_feats = np.concatenate(query_feats)
        proto_feats = np.concatenate(proto_feats)
        genproto_feats = np.concatenate(genproto_feats)

        X = np.concatenate([
            support_feats,
            query_feats,
            proto_feats,
            genproto_feats
        ])

        labels_all = (
            support_labels +
            query_labels +
            proto_labels +
            genproto_labels
        )

        types = (
            ["support"] * len(support_feats) +
            ["query"] * len(query_feats) +
            ["proto"] * len(proto_feats) +
            ["gen_proto"] * len(genproto_feats)
        )

        generate_tsne_plot(
            X,
            labels_all,
            os.path.join(
                PROTO_VIS_DIR,
                "prototype_visualization.html"
            ),
            title="Support / Query / Real Prototype / Reconstructed Prototype",
            symbols=types,
            perplexity=TSNE_PERPLEXITY,
            learning_rate=TSNE_LR
        )

        print("Prototype visualization saved to:", PROTO_VIS_DIR)


    
        # =========================================================
        # PROTOTYPE DISTRIBUTION (episodic variability)
        # =========================================================

        print("\n==== PROTOTYPE DISTRIBUTION TSNE ====\n")

        proto_points = []
        genproto_points = []
        proto_labels = []
        genproto_labels = []

        MAX_EPISODES = 100
        episode_count = 0

        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Collecting prototype distribution"):

                data = data.to(device)
                feats = model(data).view(data.size(0), -1)

                n_support = args.shot * args.test_way
                support = feats[:n_support]
                support_labels = labels[:n_support]

                support = support.reshape(args.shot, args.test_way, -1)
                support_labels = support_labels.view(args.shot, args.test_way)

                proto = support.mean(dim=0)

                episode_classes = support_labels[0]

                s = torch.stack([
                    semantic[idx_to_class[c.item()]] for c in episode_classes
                ]).to(device)

                s = s.repeat(args.shot, 1)

                gen_proto = fusion(s, support.reshape(-1, support.shape[-1]))
                gen_proto = gen_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

                for i, c in enumerate(episode_classes):
                    cls = idx_to_class[c.item()]

                    proto_points.append(proto[i].cpu().numpy())
                    genproto_points.append(gen_proto[i].cpu().numpy())

                    proto_labels.append(cls)
                    genproto_labels.append(cls)

                episode_count += 1
                if episode_count >= MAX_EPISODES:
                    break


        proto_points = np.array(proto_points)
        genproto_points = np.array(genproto_points)

        X = np.concatenate([proto_points, genproto_points])
        labels_all = proto_labels + genproto_labels
        types = ["proto"] * len(proto_points) + ["semalign"] * len(genproto_points)

        generate_tsne_plot(
            X,
            labels_all,
            os.path.join(
                PROTO_DIST_DIR,
                "prototype_distribution.html"
            ),
            title="Prototype Distribution (Episodic Variability)",
            symbols=types
        )



        # =========================================================
        # QUERY EMBEDDINGS (VISION vs SEMALIGN)
        # =========================================================

        print("\n==== QUERY EMBEDDING TSNE (VISION vs SEMALIGN) ====\n")

        vision_points = []
        semalign_points = []
        labels_all = []

        MAX_EPISODES = 50
        episode_count = 0

        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Collecting query embeddings"):

                data = data.to(device)
                feats = model(data).view(data.size(0), -1)

                n_support = args.shot * args.test_way
                support, query = feats[:n_support], feats[n_support:]
                support_labels = labels[:n_support]
                query_labels = labels[n_support:]

                # -------- vision --------
                vision_points.append(query.cpu().numpy())

                # -------- semalign --------
                episode_classes = support_labels.view(args.shot, args.test_way)[0]

                class_to_sem = {
                    idx_to_class[c.item()]: semantic[idx_to_class[c.item()]]
                    for c in episode_classes
                }

                text_query = torch.stack([
                    class_to_sem[idx_to_class[l.item()]] for l in query_labels
                ]).to(device)

                semalign_query = fusion(text_query, query)

                semalign_points.append(semalign_query.cpu().numpy())

                labels_all.extend([idx_to_class[l.item()] for l in query_labels])

                episode_count += 1
                if episode_count >= MAX_EPISODES:
                    break


        vision_points = np.concatenate(vision_points)
        semalign_points = np.concatenate(semalign_points)

        X = np.concatenate([vision_points, semalign_points])
        labels_plot = labels_all + labels_all
        types = ["vision"] * len(vision_points) + ["semalign"] * len(semalign_points)

        generate_tsne_plot(
            X,
            labels_plot,
            os.path.join(
                QUERY_DIR,
                "query_embeddings.html"
            ),
            title="Query Embeddings (Vision vs SemAlign)",
            symbols=types,
            perplexity=40,
            learning_rate=300
        )

    else:

        log.info("t-SNE desactivado (--run-tsne 0)")




# ==========================================================
# CONFUSION MATRIX ANALYSIS
# ==========================================================
#
# Generates quantitative analyses based on model predictions.
#
# Generated analyses:
#
# 1. Confusion Matrices
#    Raw prediction counts for:
#       - Proto (Visual Prototype)
#       - Gen (SemAlign Prototype)
#       - Mix (Final Prototype)
#
# 2. Normalized Confusion Matrices
#    Row-normalized versions of the confusion matrices.
#
#    Purpose:
#    Analyze per-class classification behavior independently
#    of class frequency.
#
# 3. Difference Matrices
#    Pairwise comparison between methods:
#
#       - Mix - Proto
#       - Gen - Proto
#       - Mix - Gen
#
#    Purpose:
#    Identify which class confusions are reduced or amplified
#    by semantic fusion.
#
# 4. Global Metrics
#    Accuracy, Precision, Recall and F1 scores.
#
# 5. One-vs-All Metrics
#    Class-wise Precision, Recall, Specificity and F1.
#
# 6. Top Confusions
#    Most frequent inter-class confusions.
#
# 7. Class Gain Analysis
#    Per-class accuracy improvements introduced by
#    semantic fusion.
#
# Overall Purpose:
#    Understand how semantic fusion affects classification
#    performance at both global and per-class levels.
#
# ==========================================================
    if args.run_confusion_matrix == 1:

        CONF_DIR = os.path.join(
            ANALYSIS_DIR,
            "confusion_matrix"
        )

        CONF_MATRIX_DIR = os.path.join(
            CONF_DIR,
            "matrices"
        )

        CONF_MATRIX_NORM_DIR = os.path.join(
            CONF_DIR,
            "matrices_normalized"
        )

        CONF_MATRIX_DIFF_DIR = os.path.join(
            CONF_DIR,
            "matrices_difference"
        )

        for d in [
            CONF_DIR,
            CONF_MATRIX_DIR,
            CONF_MATRIX_NORM_DIR,
            CONF_MATRIX_DIFF_DIR
        ]:
            os.makedirs(d, exist_ok=True)

        print("\n==== COMPUTING CONFUSION MATRICES ====\n")


        # BLOCK 1 : COLLECT PREDICTIONS

        all_true = []
        all_pred_proto = []
        all_pred_gen = []
        all_pred_mix = []

        with torch.no_grad():

            for data, labels in tqdm(val_loader, desc="Collecting predictions"):

                data = data.to(device)
                feats = model(data).view(data.size(0), -1)

                n_support = args.shot * args.test_way
                support, query = feats[:n_support], feats[n_support:]
                support_labels = labels[:n_support]
                query_labels = labels[n_support:]

                # -------- prototypes --------

                proto = support.reshape(args.shot, args.test_way, -1).mean(dim=0)

                s = torch.stack(
                    [semantic[idx_to_class[l.item()]] for l in support_labels]
                ).to(device)

                gen_proto = fusion(s, support).reshape(args.shot, args.test_way, -1).mean(dim=0)

                mix_proto = float(best_k) * gen_proto + (1 - float(best_k)) * proto

                # -------- predictions --------

                _, predict_proto = Cosine_classifier(proto, query)
                _, predict_gen = Cosine_classifier(gen_proto, query)
                _, predict_mix = Cosine_classifier(mix_proto, query)

                episode_classes = support_labels.view(args.shot, args.test_way)[0]

                predict_proto = predict_proto.cpu()
                predict_gen = predict_gen.cpu()
                predict_mix = predict_mix.cpu()

                real_proto = episode_classes[predict_proto].numpy()
                real_gen = episode_classes[predict_gen].numpy()
                real_mix = episode_classes[predict_mix].numpy()

                all_true.extend(query_labels.cpu().numpy())
                all_pred_proto.extend(real_proto)
                all_pred_gen.extend(real_gen)
                all_pred_mix.extend(real_mix)



        # BLOCK 2 : BUILD CONFUSION MATRICES
        all_true = np.array(all_true)
        all_pred_proto = np.array(all_pred_proto)
        all_pred_gen = np.array(all_pred_gen)
        all_pred_mix = np.array(all_pred_mix)

        class_ids = sorted(list(idx_to_class.keys()))
        class_names = [idx_to_class[c] for c in class_ids]

        cm_proto = confusion_matrix(all_true, all_pred_proto, labels=class_ids)
        cm_gen = confusion_matrix(all_true, all_pred_gen, labels=class_ids)
        cm_mix = confusion_matrix(all_true, all_pred_mix, labels=class_ids)

        cm_proto_norm = cm_proto.astype('float') / cm_proto.sum(axis=1, keepdims=True)
        cm_gen_norm = cm_gen.astype('float') / cm_gen.sum(axis=1, keepdims=True)
        cm_mix_norm = cm_mix.astype('float') / cm_mix.sum(axis=1, keepdims=True)

        cm_diff = cm_mix - cm_proto
        cm_gen_minus_proto = cm_gen - cm_proto
        cm_mix_minus_gen = cm_mix - cm_gen

        # BLOCK 3 : SAVE HTML 

        save_interactive_cm(
            cm_proto,
            class_names,
            "proto_confusion.html",
            "Proto (Visual Prototype) Confusion Matrix",
            CONF_MATRIX_DIR
        )

        save_interactive_cm(cm_gen, class_names, "gen_confusion.html", "Gen (SemAlign Output Prototype) Confusion Matrix", CONF_MATRIX_DIR)
        save_interactive_cm(cm_mix, class_names, "mix_confusion.html", "Mix (Final Prototype) Confusion Matrix", CONF_MATRIX_DIR)

        save_interactive_cm(
            cm_proto_norm,
            class_names,
            "proto_confusion_norm.html",
            "Proto (Visual Prototype) Confusion Matrix - Normalized",
            CONF_MATRIX_NORM_DIR
        )

        save_interactive_cm(cm_gen_norm, class_names, "gen_confusion_norm.html", "Gen (SemAlign Output Prototype) Confusion Matrix - Normalized", CONF_MATRIX_NORM_DIR)
        save_interactive_cm(cm_mix_norm, class_names, "mix_confusion_norm.html", "Mix (Final Prototype) Confusion Matrix - Normalized", CONF_MATRIX_NORM_DIR)

        save_interactive_cm(
            cm_diff, 
            class_names,
            "mix_minus_proto.html", 
            "Mix (Final Prototype) - Proto (Visual Prototype) Improvement Matrix",
            CONF_MATRIX_DIFF_DIR
        )

        save_interactive_cm(
            cm_gen_minus_proto,
            class_names,
            "gen_minus_proto.html",
            "Gen (SemAlign Output) - Proto (Visual Prototype) Improvement Matrix",
            CONF_MATRIX_DIFF_DIR
        )

        save_interactive_cm(
            cm_mix_minus_gen,
            class_names,
            "mix_minus_gen.html",
            "Mix (Final Prototype) - Gen (SemAlign Output) Improvement Matrix",
            CONF_MATRIX_DIFF_DIR
        )


        # BLOCK 4 : BUILD DATAFRAMES

        df_cm_proto = pd.DataFrame(cm_proto, index=class_names, columns=class_names)
        df_cm_gen = pd.DataFrame(cm_gen, index=class_names, columns=class_names)
        df_cm_mix = pd.DataFrame(cm_mix, index=class_names, columns=class_names)
        df_cm_diff = pd.DataFrame(cm_diff, index=class_names, columns=class_names)
        df_cm_gen_minus_proto = pd.DataFrame(cm_gen_minus_proto, index=class_names, columns=class_names)
        df_cm_mix_minus_gen = pd.DataFrame(cm_mix_minus_gen, index=class_names, columns=class_names)

        # BLOCK 5 : COMPUTE STATISTICAL ANALYSES

        metrics_proto = compute_metrics(all_true, all_pred_proto)
        metrics_gen = compute_metrics(all_true, all_pred_gen)
        metrics_mix = compute_metrics(all_true, all_pred_mix)

        df_global = pd.DataFrame([
            {"model": "proto", **metrics_proto},
            {"model": "gen", **metrics_gen},
            {"model": "mix", **metrics_mix}
        ])

        df_proto_ova = compute_ova(cm_proto, class_names)
        df_gen_ova = compute_ova(cm_gen, class_names)
        df_mix_ova = compute_ova(cm_mix, class_names)


        df_top_proto = top_confusions(cm_proto, class_names)
        df_top_gen = top_confusions(cm_gen, class_names)
        df_top_mix = top_confusions(cm_mix, class_names)


        # ================= CLASS GAIN ANALYSIS =================

        df_acc_proto = per_class_accuracy(cm_proto, class_names)
        df_acc_gen = per_class_accuracy(cm_gen, class_names)
        df_acc_mix = per_class_accuracy(cm_mix, class_names)

        df_gain = df_acc_proto.copy()
        df_gain.rename(columns={"accuracy": "proto_accuracy"}, inplace=True)

        df_gain["gen_accuracy"] = df_acc_gen["accuracy"]
        df_gain["mix_accuracy"] = df_acc_mix["accuracy"]

        df_gain["gain_gen_vs_proto"] = df_gain["gen_accuracy"] - df_gain["proto_accuracy"]
        df_gain["gain_mix_vs_proto"] = df_gain["mix_accuracy"] - df_gain["proto_accuracy"]
        df_gain["gain_mix_vs_gen"] = df_gain["mix_accuracy"] - df_gain["gen_accuracy"]

        df_gain = df_gain.sort_values("gain_mix_vs_proto", ascending=False)

        # ================= SAVE EXCEL =================

        excel_matrix_path = os.path.join(CONF_DIR, "confusion_matrices.xlsx")

        # xlsx with confusion matrix
        with pd.ExcelWriter(excel_matrix_path) as writer:


            df_cm_proto.to_excel(writer, sheet_name="Confusion_Proto")
            df_cm_gen.to_excel(writer, sheet_name="Confusion_Gen")
            df_cm_mix.to_excel(writer, sheet_name="Confusion_Mix")
            df_cm_diff.to_excel(writer, sheet_name="Mix_minus_Proto")
            df_cm_gen_minus_proto.to_excel(writer, sheet_name="Gen_minus_Proto")
            df_cm_mix_minus_gen.to_excel(writer, sheet_name="Mix_minus_Gen")

  
        excel_stats_path = os.path.join(CONF_DIR, "statistics_analysis.xlsx")

        # xlsx with stadistics 
        with pd.ExcelWriter(excel_stats_path) as writer:

            df_global.to_excel(writer, sheet_name="Global Metrics", index=False)

            df_proto_ova.to_excel(writer, sheet_name="Proto_OvA", index=False)
            df_gen_ova.to_excel(writer, sheet_name="Gen_OvA", index=False)
            df_mix_ova.to_excel(writer, sheet_name="Mix_OvA", index=False)

            # ===== Top Confusions =====

            df_top_proto.to_excel(writer, sheet_name="Top_confusions_proto", index=False)
            df_top_gen.to_excel(writer, sheet_name="Top_confusions_gen", index=False)
            df_top_mix.to_excel(writer, sheet_name="Top_confusions_mix", index=False)

            # ===== Per-Class Accuracy =====

            df_acc_proto.to_excel(writer, sheet_name="Accuracy_proto", index=False)
            df_acc_gen.to_excel(writer, sheet_name="Accuracy_gen", index=False)
            df_acc_mix.to_excel(writer, sheet_name="Accuracy_mix", index=False)

            # ===== Gain Analysis =====

            df_gain.to_excel(writer, sheet_name="Class_gain_analysis", index=False)

        print("\nConfusion matrices saved to:", CONF_DIR)
        print("Matrix Excel saved to:", excel_matrix_path)
        print("Statistics Excel saved to:", excel_stats_path)

    else:

        log.info("confusion matrix no ejecutado (flag --run_confusion_matrix 0)")



# ==========================================================
# HIERARCHICAL CLUSTERING ANALYSIS
# ==========================================================
#
# Generates a class hierarchy based on mutual confusion.
#
# Method:
#   1. Build a normalized confusion matrix.
#   2. Compute mutual confusion between class pairs.
#   3. Convert confusion similarity into a distance metric.
#   4. Perform hierarchical clustering.
#   5. Visualize the resulting dendrogram.
#
# Purpose:
#   Identify groups of classes that are frequently confused
#   by the final SemAlign model.
#
# ==========================================================
    if args.run_hierarchical == 1:

        if args.run_confusion_matrix == 0:
            raise ValueError(
                "Hierarchical clustering requires confusion matrices. Currenlty set at 0"
            )

        print("\n==== HIERARCHICAL CLUSTERING ====\n")

        CLUSTER_DIR = os.path.join(
            ANALYSIS_DIR,
            "hierarchical_clustering"
        )

        os.makedirs(CLUSTER_DIR, exist_ok=True)

        # BUILD DISTANCE MATRIX
        mix_cm_distance = build_mutual_confusion_distance(
            cm_mix_norm
        )

        # CONVERT TO CONDENSED FORM

        mix_cm_condensed = squareform(
            mix_cm_distance
        )

        # HIERARCHICAL CLUSTERING

        mix_cm_linkage = linkage(
            mix_cm_condensed,
            method="average"
        )

        # SAVE DENDROGRAM
        save_dendrogram(
            mix_cm_linkage,
            class_names,
            "Final Model Confusion Dendrogram",
            os.path.join(
                CLUSTER_DIR,
                "mix_confusion_dendrogram.png"
            )
        )

        print(
            "Hierarchical clustering saved to:",
            CLUSTER_DIR
        )

    else:

        log.info(
            "hierarchical clustering disabled "
            "(--run-hierarchical 0)"
        )



    print("\n\nFin del codigo!")



# results/

# ├── results.txt
# ├── results.xlsx
# │
# ├── confusion_matrix/
# │   ├── matrices/
# │   ├── matrices_normalized/
# │   ├── matrices_difference/
# │   ├── confusion_matrices.xlsx
# │   └── statistics_analysis.xlsx
# │
# ├── tsne/
# │   ├── embeddings/
# │   │   ├── text_encoder.html
# │   │   ├── vision_encoder.html
# │   │   ├── semalign_encoder.html
# │   │   └── vision_vs_semalign.html
# │   │
# │   ├── prototype_visualization/
# │   │   └── prototype_visualization.html
# │   │
# │   ├── prototype_distribution/
# │   │   └── prototype_distribution.html
# │   │
# │   └── query_embeddings/
# │       └── query_embeddings.html
# │
# └── hierarchical_clustering/
#     └── mix_confusion_dendrogram.png