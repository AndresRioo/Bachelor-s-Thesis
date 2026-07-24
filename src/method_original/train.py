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
    parser.add_argument('--exp-tag', type=str, default='baseline')
    args = parser.parse_args()
    



    if args.backbone == 'resnet':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)
        
    args.work_dir = '{}_{}_{}_{}_{}_{}'.format(args.backbone, args.dataset, args.mode, args.text_type, args.center, args.shot)


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

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
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
            name=f"baseline_{today}_{args.work_dir}",
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
    
    for epoch in range(args.max_epoch):

        epoch_recon_loss = []
        epoch_total_loss = []

        for step, [data, labels] in enumerate(tqdm(train_loader)):
            proto = torch.tensor(np.array([proto_center[idx_to_class[l.item()]] for l in labels])).to(device)
            text_feature = torch.stack([semantic[idx_to_class[l.item()]] for l in labels]).to(device)
            with torch.no_grad():
                img_feature = model(data.to(device))

            optimizer.zero_grad()
            H.zero_grad()
            H.train()
            fusion = H(text_feature, img_feature)
            recon_loss = F.l1_loss(fusion, proto)

            g_loss = recon_loss
            g_loss.backward()
            optimizer.step()

            # estadisticas wandb
            epoch_recon_loss.append(recon_loss.item())
            epoch_total_loss.append(g_loss.item())

        log.info(
            "[Epoch %d/%d] [recon loss: %f] "
            % (epoch, args.max_epoch, recon_loss.item(),)
        )

        mean_recon = np.mean(epoch_recon_loss)
        mean_total_loss = np.mean(epoch_total_loss)


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