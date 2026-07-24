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
    parser.add_argument('--exp-tag', type=str, default='hierarchical_adaptive_radius')


    # ARGUMENTOS NUEVOS PARA EL CLUSTERING
    parser.add_argument(
        '--cluster-level',
        type=int,
        default=16
    )

    parser.add_argument(
        '--radius-scale',
        type=float,
        default=0.2
    )

    parser.add_argument(
        '--cluster-lambda',
        type=float,
        default=0.01
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

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

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
    wandb.init(
        project="semfew-tfg",
        name=f"hc_margen_{today}_{args.work_dir}",
        tags=[
            args.dataset,
            args.backbone,
            f"{args.shot}-shot",
            args.mode,
            args.exp_tag,
            f"cluster-{args.cluster_level}",
            f"radius_scale-{args.radius_scale}",
            f"cluster_lambda-{args.cluster_lambda}",
            f"cluster_mode-{args.cluster_mode}",
            EXPERIMENT_TAG
        ],
        config=vars(args),
        group=f"{args.dataset}_{args.shot}shot"
    )





        
    H = SemAlign(feat_size, args.semantic_size, h_size=4096, drop=args.drop).to(device)
    optimizer = torch.optim.Adam(H.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    if 'ImageNet' in args.dataset:
        semantic = torch.load('./semantic/imagenet_semantic_{}_{}.pth'.format(args.mode, args.text_type), weights_only=False)['semantic_feature']
    else:
        semantic = torch.load('./semantic/cifar100_semantic_{}_{}.pth'.format(args.mode, args.text_type), weights_only=False)['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}

    # Cargar clusters 

    args.cluster_file = os.path.join(
        BASE_DIR,
        './precomputed_clusters',
        f'clusters_{args.dataset}_{args.backbone}_{args.cluster_mode}.pth'
    )

    if not os.path.exists(args.cluster_file):
        raise ValueError(
            f'Cluster file not found: {args.cluster_file}'
        )

    cluster_data = torch.load(
        args.cluster_file,
        weights_only=False
    )

    level_data = cluster_data['levels'][args.cluster_level]

    class_to_cluster = level_data['class_to_cluster']

    cluster_centers = level_data['cluster_centers']

    cluster_radii = level_data['cluster_radii']

    print(f'Loaded cluster level: {args.cluster_level}')
    print(f'Num clusters: {len(cluster_centers)}')

    gap_acc = -1
    max_acc1 = 0.0

    
    

    # train loop
    for epoch in range(args.max_epoch):

        epoch_recon_loss = []
        epoch_cluster_loss = []
        epoch_total_loss = []
        epoch_cluster_distance = []
        epoch_active_constraints = []
        epoch_radius = []
        epoch_danger_radius = []
        epoch_mean_violation = []
        epoch_max_violation = []
        all_normalized_distances = []

        for step, [data, labels] in enumerate(tqdm(train_loader)):
            proto = torch.tensor(np.array([proto_center[idx_to_class[l.item()]] for l in labels])).to(device)
            text_feature = torch.stack([semantic[idx_to_class[l.item()]] for l in labels]).to(device)
            with torch.no_grad():
                img_feature = model(data.to(device))

            optimizer.zero_grad()
            H.zero_grad()
            H.train()


            fusion = H(text_feature, img_feature)

            # Loss original paper (H1 entre proto clase y semalign)
            recon_loss = F.l1_loss(fusion, proto)



            # lista donde guardaremos:
            # el centro del cluster al que pertenece cada clase
            # el radio del cluster al que pertenece cada clase
            #
            # tamaño final:
            # len = batch_size
            batch_cluster_centers = []
            batch_cluster_radii = []
            valid_indices = []

            for i, l in enumerate(labels):

                # de label numerico a clase
                class_name = idx_to_class[l.item()]

                # Algunas clases no pertenecen a ningun cluster porque
                # fueron eliminadas al aplicar MIN_CLUSTER_SIZE.
                #
                # En ese caso simplemente NO participan en la
                # regularizacion de clusters.
                if class_name not in class_to_cluster:
                    continue

                # cluster de la clase
                cluster_id = class_to_cluster[class_name]

                # centro geometrico del cluster
                cluster_center = cluster_centers[cluster_id]

                # obtener radio REAL del cluster, precalculado ya
                cluster_radius = cluster_radii[cluster_id]

                batch_cluster_centers.append(cluster_center)
                batch_cluster_radii.append(cluster_radius)

                # guardar posicion dentro del batch
                valid_indices.append(i)

            # Si ninguna clase del batch pertenece a un cluster,
            # la regularizacion debe ser 0
            if len(batch_cluster_centers) == 0:

                cluster_loss = torch.tensor(
                    0.0,
                    device=device
                )

                cosine_distance = torch.tensor(
                    [0.0],
                    device=device
                )

                danger_radius = torch.tensor(
                    [0.0],
                    device=device
                )

                batch_cluster_radii = torch.tensor(
                    [0.0],
                    device=device
                )

            else:

                # pasar a tensor
                batch_cluster_radii = torch.tensor(
                    batch_cluster_radii,
                    device=device
                )

                # expandir radio original con el parametro de margen
                # si cae dentro se penaliza
                danger_radius = (
                    batch_cluster_radii *
                    (1.0 + args.radius_scale)
                )

                # Solo usamos muestras que realmente tienen cluster
                fusion_valid = fusion[valid_indices]

                # normalizar L2 para cosine
                fusion_norm = F.normalize(
                    fusion_valid,
                    dim=1
                )

                batch_cluster_centers = torch.stack(
                    batch_cluster_centers
                ).to(device)

                cluster_center_norm = F.normalize(
                    batch_cluster_centers,
                    dim=1
                )

                # cosine similarity

                # similitud entre:
                # embedding generado por SemAlign
                # centro del cluster correspondiente
                cosine_sim = torch.sum(
                    fusion_norm * cluster_center_norm,
                    dim=1
                )

                # pasar a distancia
                cosine_distance = 1.0 - cosine_sim

                # normalizar para el histograma
                normalized_distance = (
                    cosine_distance /
                    (danger_radius + 1e-8)
                )

                all_normalized_distances.extend(
                    normalized_distance.detach().cpu().numpy()
                )

                # penalizar solo si distance < danger_radius
                cluster_loss = torch.relu(
                    danger_radius - cosine_distance
                ).mean()


            # TOTAL LOSS
            g_loss = (
                recon_loss +
                args.cluster_lambda * cluster_loss
            )

            g_loss.backward()
            optimizer.step()
            


            # METRICAS


            # porcentaje de samples penalizados
            active_constraints = (
                cosine_distance < danger_radius
            ).float().mean().item()

            # radio medio real del cluster
            mean_radius = batch_cluster_radii.mean().item()

            # radio final tras añadir margen
            mean_danger_radius = danger_radius.mean().item()

            # cuanto se viola el margen (EVITAR VALORES NEGATIVOS, indicaria que esta fuera)
            margin_violation = torch.relu(
                danger_radius - cosine_distance
            )

            mean_violation = margin_violation.mean().item()

            max_violation = margin_violation.max().item()

            # estadisticas wandb
            epoch_recon_loss.append(recon_loss.item())
            epoch_cluster_loss.append(cluster_loss.item())
            epoch_total_loss.append(g_loss.item())
            epoch_active_constraints.append(active_constraints)
            epoch_radius.append(mean_radius)
            epoch_danger_radius.append(mean_danger_radius)
            epoch_mean_violation.append(mean_violation)
            epoch_max_violation.append(max_violation)
            epoch_cluster_distance.append(cosine_distance.mean().item())



        log.info(
            "[Epoch %d/%d] [recon loss: %f] "
            % (epoch, args.max_epoch, recon_loss.item(),)
        )

        mean_recon = np.mean(epoch_recon_loss)
        mean_cluster = np.mean(epoch_cluster_loss)  
        mean_total = np.mean(epoch_total_loss)
        mean_cluster_distance = np.mean(epoch_cluster_distance)

        mean_active_constraints = np.mean(epoch_active_constraints)

        mean_radius_epoch = np.mean(epoch_radius)

        mean_danger_radius_epoch = np.mean(
            epoch_danger_radius
        )

        mean_violation_epoch = np.mean(
            epoch_mean_violation
        )

        max_violation_epoch = np.max(
            epoch_max_violation
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
                "train/total_loss": mean_total,
                "train/cluster_loss": mean_cluster,
                "train/cluster_distance": mean_cluster_distance,
                "train/loss_ratio": cluster_loss.item() / (recon_loss.item() + 1e-8),

                "train/active_constraints": mean_active_constraints,
                "train/cluster_radius": mean_radius_epoch,
                "train/danger_radius": mean_danger_radius_epoch,
                "train/mean_violation": mean_violation_epoch,
                "train/max_violation": max_violation_epoch,

                "train/normalized_distance_histogram":
                    wandb.Histogram(all_normalized_distances),

                                
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
                "epoch": epoch
            })

    writer.close()
    wandb.finish()


# EXPLICACION DE CADA VALOR NUEVO

# active_constraints : cuánto actúa realmente el regularizador.

# 0.02 -> casi nunca penaliza
# 0.50 -> actúa bastante
# 0.95 -> está empujando todo

# cluster_radius : Radio real medio de los clusters. comparar visual vs semantic

# danger_radius : Radio final tras añadir margen. interpretar el parametro 'radius_scale'.

# mean_violation : cuánto se están metiendo dentro de la zona prohibida. Media de : (r+m)−d

# max_violation : detectar explosiones.

# histograma normalized distances : indica si las muestras 

# - normalized_distance = 1 , justo en el borde
# - normalized_distance < 1 , está siendo penalizada
# - normalized_distance > 1 , está fuera de la zona prohibida


# TFG:

# active_constraints
# mean_violation
# max_violation