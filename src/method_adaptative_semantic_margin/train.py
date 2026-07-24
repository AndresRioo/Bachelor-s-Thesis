import os
import sys



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from method_original.SemAlign import SemAlign

from data.samplers import CategoriesSampler
# from data.tiered_imagenet import tieredImageNet
from logger import loggers
from model.res12 import Res12
from model.swin_transformer import swin_tiny
from utils import Cosine_classifier, count_95acc, transform_val_cifar, transform_train_cifar, \
    transform_train, count_kacc, transform_val
from utils import transform_val_224_cifar, transform_train_224_cifar, transform_train_224, transform_val_224

import random
import numpy as np
import torch

import wandb


seed = 112
os.environ["PYTHONHASHSEED"] = str(seed)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--test-batch', type=int, default=600)
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--feat-size', type=int, default=640)
    parser.add_argument('--semantic-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'bert'])
    parser.add_argument('--text_type', type=str, default='gpt',
                        choices=['gpt', 'name', 'definition'])
    parser.add_argument('--dataset', type=str, default='TieredImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'swin'])
    parser.add_argument('--train-episodes', type=int, default=600)


    parser.add_argument('--exp-tag', type=str, default='adaptative_s_margin')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda-margin', type=float, dest='lambda_margin', default=0.1)

    args = parser.parse_args()
    



    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)
        
    args.work_dir = 'adaptative_margin_{}_{}_{}_{}_{}_{}_alpha{}_lambda{}'.format(
        args.backbone,
        args.dataset,
        args.mode,
        args.text_type,
        args.center,
        args.shot,
        args.alpha,
        args.lambda_margin
    )

    if args.dataset == 'TieredImageNet':
        args.num_workers = 0

    if os.path.exists(args.work_dir) is False:
        os.mkdir(args.work_dir)

    log = loggers(os.path.join(args.work_dir, 'train'))
    log.info(vars(args))

    writer = SummaryWriter(os.path.join(args.work_dir, 'train_semantic'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'MiniImageNet':
        args.val = 'MINIIMAGENET/miniimagenet/val'
        args.train = 'MINIIMAGENET/miniimagenet/train'
        train_dataset = ImageFolder(args.train, transform=transform_train if args.backbone == 'resnet' else transform_train_224)
        val_dataset = ImageFolder(args.val, transform=transform_val if args.backbone == 'resnet' else transform_val_224)

    elif args.dataset == 'FC100':
        args.val = '/path/to/your/fc100/val'
        args.train = '/path/to/your/fc100/train'
        train_dataset = ImageFolder(args.train, transform=transform_train_cifar if args.backbone == 'resnet' else transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar)

    elif args.dataset == 'CIFAR-FS':
        args.val = 'CIFAR-FS/cifar-fs/val'
        args.train = 'CIFAR-FS/cifar-fs/train'
        train_dataset = ImageFolder(args.train, transform=transform_train_cifar if args.backbone == 'resnet' else transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=transform_val_cifar if args.backbone == 'resnet' else transform_train_224_cifar)

    elif args.dataset == 'TieredImageNet':
        train_dataset = tieredImageNet(setname='train', augment=True)
        val_dataset = tieredImageNet(setname='val')

        if args.backbone == 'swin':
            args.val = '/path/to/your/tiredimagenet/val'
            args.train = '/path/to/your/tiredimagenet/train'
            train_dataset = ImageFolder(args.train, transform=transform_train_224)
            val_dataset = ImageFolder(args.val, transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    val_idx_to_class = val_dataset.class_to_idx
    val_idx_to_class = {k: v for v, k in val_idx_to_class.items()}
    idx_to_class = train_dataset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}


    # Cambios para fijar la semilla (cambiar train_loader)
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    val_sampler = CategoriesSampler(val_dataset.targets, args.test_batch, args.test_way, args.shot + args.query)


    
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)
    
    if args.backbone == 'resnet':
        proto_center = torch.load('center_{}_{}.pth'.format(args.dataset, args.backbone), weights_only=False)[args.center]
    elif args.backbone == 'swin':
        proto_center = torch.load('center_{}_{}.pth'.format(args.dataset, args.backbone), weights_only=False)[args.center]
     
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
    
    if args.backbone == 'resnet':
        feat_size = 640
    elif args.backbone == 'swin':
        feat_size = 768


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

    # METRICAS DE TRAIN 
    try:

        print("Conectando con wandb...")

        wandb.init(
            project="semfew-tfg",
            name=f"adaptative_margin_{today}_{args.work_dir}",
            tags=[
                args.dataset,
                args.backbone,
                f"{args.shot}-shot",
                args.mode,
                args.exp_tag,
                f"alpha_{args.alpha}",
                f"lambda_{args.lambda_margin}",
                EXPERIMENT_TAG
            ],
            config=vars(args),
            group=f"{args.dataset}_{args.shot}shot",
            settings=wandb.Settings(
                    init_timeout=900,
                    start_method="thread"
                ),
            reinit=True
        )

        print("Conectado!")

    except Exception as e:
        
        print(f"\n\n\nWandB online failed: {e}\n\n\n")

        wandb.init(
            project="semfew-tfg",
            name=f"baseline_{today}_{args.work_dir}",
            mode="offline",
            tags=[
                args.dataset,
                args.backbone,
                f"{args.shot}-shot",
                args.mode,
                args.exp_tag,
                EXPERIMENT_TAG
            ],
            config=vars(args),
            group=f"{args.dataset}_{args.shot}shot",
            settings=wandb.Settings(
                    init_timeout=900,
                    start_method="thread"
                ),
            reinit=True
        )




        
    H = SemAlign(feat_size, args.semantic_size, h_size=4096, drop=args.drop).to(device)
    optimizer = torch.optim.Adam(H.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    if 'ImageNet' in args.dataset:
        semantic = torch.load('./semantic/imagenet_semantic_{}_{}.pth'.format(args.mode, args.text_type), weights_only=False)['semantic_feature']
    else:
        semantic = torch.load('./semantic/cifar100_semantic_{}_{}.pth'.format(args.mode, args.text_type), weights_only=False)['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}

    gap_acc = -1
    max_acc1 = 0.0


    # cargar matriz de distancias
    tree_dist = np.load('semantic_tree_dist.npy')

    # mapping global
    class_to_idx_global = {
        cls: i for i, cls in enumerate(sorted(semantic.keys()))
    }

    print("Adaptive margin loss enabled")


    # Hacer training episodico 
    train_sampler = CategoriesSampler(
        train_dataset.targets,
        args.train_episodes,
        args.test_way,
        args.shot + args.query
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    
    for epoch in range(args.max_epoch):

        epoch_recon_loss = []
        epoch_total_loss = []
        epoch_margin_loss = []

        epoch_loss_ratio = []
        epoch_violation_rate = []
        epoch_margin_gap = []
        epoch_mean_violation = []

        epoch_semantic_visual_corr = []

        for step, (data, labels) in enumerate(tqdm(train_loader)):

            data = data.to(device)

            with torch.no_grad():
                features = model(data)

            optimizer.zero_grad()
            H.train()

            # EPISODE SPLIT
            n_support = args.shot * args.test_way

            support = features[:n_support]

            support_labels = labels[:n_support].to(device)

            # SEMANTIC FEATURES
            s_support = torch.stack([
                semantic[idx_to_class[l.item()]]
                for l in support_labels
            ]).to(device)

            # SEMALIGN
            support_embed = H(s_support, support)

            # RECON LOSS
            proto_center_batch = torch.tensor(np.array([
                proto_center[idx_to_class[l.item()]]
                for l in support_labels
            ])).to(device)

            recon_loss = F.l1_loss(
                support_embed,
                proto_center_batch
            )

            # =========================================
            # TOTAL LOSS INIT
            # =========================================

            total_loss = recon_loss

            margin_loss = torch.tensor(0.0).to(device)

            # ADAPTIVE MARGIN LOSS
            support_embed_reshape = support_embed.view(
                args.shot,
                args.test_way,
                -1
            )

            # Calcula un prototipo medio por clase dentro del episodio.
            prototypes = support_embed_reshape.mean(dim=0)

            # Normaliza los prototipos para usar distancia coseno.
            proto_norm = F.normalize(prototypes, dim=1)

            # Calcula la matriz de distancias coseno entre prototipos.
            dist_matrix = 1 - torch.matmul(
                proto_norm,
                proto_norm.T
            )

            # Obtiene los nombres de las clases presentes en el episodio.
            class_names = [
                idx_to_class[support_labels[i].item()]
                for i in range(args.test_way)
            ]

            # Convierte los nombres de clase a índices globales.
            idxs = [
                class_to_idx_global[c]
                for c in class_names
            ]

            # Extrae la submatriz semántica correspondiente al episodio actual.
            tree_sub = torch.tensor(
                tree_dist[np.ix_(idxs, idxs)]
            ).float().to(device)

            # Normaliza las distancias semánticas al rango [0,1].
            tree_sub = tree_sub / tree_sub.max()

            # Construye márgenes adaptativos según distancia semántica.
            margin_matrix = args.alpha * (1 - tree_sub)

            # Crea una máscara para usar solo pares únicos sin diagonal.
            mask = torch.triu(
                torch.ones(
                    args.test_way,
                    args.test_way,
                    device=device
                ),
                diagonal=1
            )

            # Calcula cuánto viola cada par el margen impuesto.
            violation_matrix = F.relu(
                margin_matrix - dist_matrix
            )

            # Adaptive margin loss
            margin_loss = (
                (violation_matrix * mask).sum()
                / mask.sum()
            )

            # Combina reconstruction loss + adaptive margin loss
            total_loss = (
                recon_loss
                + args.lambda_margin * margin_loss
            )

            # Calcular ratio entre losses para comparar y ver cual domina 
            margin_contrib = args.lambda_margin * margin_loss

            loss_ratio = (
                margin_contrib.detach()
                / (recon_loss.detach() + 1e-8)
            )

            # Calcular violation rate #(dij​<mij​)​ / N pares  - 1 : todo colapsado, 0 margen satisfecho
            violations = (
                (margin_matrix - dist_matrix) > 0
            ).float()

            violation_rate = (
                (violations * mask).sum()
                / mask.sum()
            )

            # MARGIN GAP
            # d_ij - m_ij
            # positivo = margen satisfecho
            # negativo = violación
            margin_gap = (
                ((dist_matrix - margin_matrix) * mask).sum()
                / mask.sum()
            )

            # VIOLATION MAGNITUDE
            # cuanto se viola el margen
            mean_violation = (
                (violation_matrix * mask).sum()
                / ((violations * mask).sum() + 1e-8)
            )

            # correlación semantico visual
            tree_flat = tree_sub.detach().cpu().numpy().flatten()

            dist_flat = (
                dist_matrix.detach()
                .cpu()
                .numpy()
                .flatten()
            )

            # evitar la diagonal 
            pair_mask = torch.triu(
                torch.ones_like(tree_sub),
                diagonal=1
            ).bool()

            tree_flat = tree_sub[pair_mask].detach().cpu().numpy()
            dist_flat = dist_matrix[pair_mask].detach().cpu().numpy()

            corr_matrix = np.corrcoef(
                tree_flat,
                dist_flat
            )

            semantic_visual_corr = corr_matrix[0, 1]

            if np.isnan(semantic_visual_corr):
                semantic_visual_corr = 0.0

            epoch_semantic_visual_corr.append(
                semantic_visual_corr
            )


            total_loss.backward()
            optimizer.step()

            epoch_recon_loss.append(recon_loss.item())
            epoch_total_loss.append(total_loss.item())
            epoch_margin_loss.append(margin_loss.item())

            epoch_loss_ratio.append(loss_ratio.item())
            epoch_violation_rate.append(violation_rate.item())
            epoch_margin_gap.append(margin_gap.item())
            epoch_mean_violation.append(mean_violation.item())

        log.info(
            "[Epoch %d/%d] [recon loss: %f] "
            % (epoch, args.max_epoch, recon_loss.item(),)
        )

        mean_recon = np.mean(epoch_recon_loss)
        mean_total_loss = np.mean(epoch_total_loss)
        mean_margin_loss = np.mean(epoch_margin_loss)

        mean_ratio = np.mean(epoch_loss_ratio)
        mean_violation_rate = np.mean(epoch_violation_rate)
        mean_margin_gap = np.mean(epoch_margin_gap)
        mean_mean_violation = np.mean(epoch_mean_violation)

        mean_semantic_visual_corr = np.mean(
            epoch_semantic_visual_corr
        )
                
        lr_scheduler.step()

        # val
        ks = np.arange(0, 101) * 0.01
        label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)
        if epoch % 10 == 0 or epoch > 0:
            P_acc = {}
            O_acc = []
            G_acc = []

            proto_cos_list = []
            proto_l2_list = []


            with torch.no_grad():
                for data, labels in tqdm(val_loader):
                    data = data.to(device)
                    data = model(data).view(data.size(0), -1)
                    n_support = args.shot * args.test_way
                    support, query = data[:n_support], data[n_support:]

                    proto = support.reshape(args.shot, args.test_way, -1).mean(dim=0)
                    s = torch.stack([semantic[val_idx_to_class[l.item()]] for l in labels[:n_support]]).to(device)
                    gen_proto = H(s, support)
                    gen_proto = gen_proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
                    
                    # Prototype alignment metrics
                    proto_cos = F.cosine_similarity(
                        gen_proto,
                        proto,
                        dim=-1
                    ).mean()

                    proto_l2 = torch.norm(
                        gen_proto - proto,
                        dim=-1
                    ).mean()

                    proto_cos_list.append(proto_cos.item())
                    proto_l2_list.append(proto_l2.item())

                    _, predict0 = Cosine_classifier(proto, query)
                    _, predict1 = Cosine_classifier(gen_proto, query)
                    O_acc.append(((predict0 == label).sum() / len(label)).item())
                    G_acc.append(((predict1 == label).sum() / len(label)).item())
                    
                    for f in ks:
                        if str(f) in P_acc:
                            P_acc[str(f)].append(count_kacc(proto, gen_proto, query, f, args))
                        else:
                            P_acc[str(f)] = []
                            P_acc[str(f)].append(count_kacc(proto, gen_proto, query, f, args))

                O_acc, O_95 = count_95acc(np.array(O_acc))
                G_acc, G_95 = count_95acc(np.array(G_acc))

                mean_proto_cos = np.mean(proto_cos_list)
                mean_proto_l2 = np.mean(proto_l2_list)

                max_acc = {
                    'k': 0,
                    'acc': 0,
                    'acc95': 0,
                }
                for k, v in P_acc.items():
                    P_acc[k] = count_95acc(np.array(v))
                    
                    wandb.log({
                        f"k_curve/acc_k_{k}": P_acc[k][0] * 100
                    }, step=epoch)
                    
                    if P_acc[k][0] > max_acc['acc']:
                        max_acc['acc'] = P_acc[k][0]
                        max_acc['acc95'] = P_acc[k][1]
                        max_acc['k'] = k
                cur_gap = max_acc['acc'] - O_acc
                if cur_gap > gap_acc:
                    gap_acc = cur_gap
                if max_acc['acc'] > max_acc1:
                    max_acc1 = max_acc['acc']
                    torch.save({
                        'G': H,
                        'epoch': epoch,
                        'k': max_acc['k'],
                        'acc': max_acc1
                    }, os.path.join(args.work_dir, 'epoch_best.pth'.format(epoch)))
                    log.info('best epoch: %d' % epoch)
                    print('save', epoch)
                    wandb.run.summary["best_mix_acc"] = max_acc1 * 100

            writer.add_scalars("add_scalars/acc", {'origin': O_acc,
                                                   'complete': max_acc['acc']}, epoch)
            log.info(
                'epoch: %d |origin acc: %.2f+%.2f%% |complete acc: %.2f+%.2f%% |gap: %.2f/%.2f |k: %s' % (
                    epoch, O_acc * 100, O_95 * 100, max_acc['acc'] * 100, max_acc['acc95'] * 100,
                    gap_acc * 100, cur_gap * 100, max_acc['k']))
            log.info('ACC |proto acc: %.2f+%.2f%% |gen acc: %.2f+%.2f%% |Max: %.2f' % (
                O_acc * 100, O_95 * 100, G_acc * 100, G_95 * 100, 100 * max_acc1))
            
            wandb.log({

                # TRAIN
                "train/recon_loss": mean_recon,
                "train/total_loss": mean_total_loss,
                "train/margin_loss": mean_margin_loss,

                "train/loss_ratio": mean_ratio,
                "train/violation_rate": mean_violation_rate,
                "train/margin_gap": mean_margin_gap,
                "train/mean_violation": mean_mean_violation,

                "train/semantic_visual_corr":    mean_semantic_visual_corr,

                # VALIDATION
                "val/proto_acc": O_acc * 100,
                "val/gen_acc": G_acc * 100,
                "val/mix_acc": max_acc['acc'] * 100,

                "val/proto_acc_95": O_95 * 100,
                "val/gen_acc_95": G_95 * 100,
                "val/mix_acc_95": max_acc['acc95'] * 100,

                "val/proto_cosine": mean_proto_cos,
                "val/proto_l2": mean_proto_l2,

                "val/best_k": float(max_acc['k']),
                "val/gap": cur_gap * 100,

                # META
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "train/lambda_h": args.lambda_margin,
                "train/alpha": args.alpha,
            })

    writer.close()
    wandb.finish()


# DEFINICION DE LAS GRAFICAS

# loss_ratio : quién domina el gradiente.
    
# violation_rate : # (Dij < mij ) / N pares

#   1 : todo viola
#   0 : nada entra en el margen

# margin_gap : dij​−mij​  
# muestra si la geometria va mejorando o no

# mean_violation : cuanto falla 


# correlacion entre tree sub y distance matrix
# Interpretación:

# 0 - la geometría visual ignora la semántica.
# 1 - las distancias visuales reproducen perfectamente la estructura semántica.

# regimen 1 Los prototipos empiezan a organizarse siguiendo la estructura semántica
# epoch 0  -> 0.10
# epoch 10 -> 0.28
# epoch 20 -> 0.41
# epoch 30 -> 0.50

# regimen 2 : regularizador apenas está afectando a la geometría
# 0.05
# 0.04
# 0.06
# 0.03

# regimen 3 : distancia semántica está dominando completamente la organización del embedding.
# 0.80