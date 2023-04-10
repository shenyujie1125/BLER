import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from transforms import *
from logger import setup_logging
from model import loss
from trainer.trainer import Trainer
from dataset import TSNDataSet
from model.models import TSN

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(args, config):
    if args.modality == 'RGB':
        data_length = 1
    else:
        data_length = None
    if args.modality_1 == 'Flow':
        data_length_1 = 5
    else:
        data_length_1 = None

    model = TSN(26, args.num_segments, args.modality, args.modality_1,
                base_model=args.arch, new_length=data_length, new_length_1=data_length_1, embed=args.embed,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn,
                context=args.context, args=args)

    input_mean = model.input_mean
    input_std = model.input_std
    input_mean_1 = model.input_mean_1
    input_std_1 = model.input_std_1

    policies = model.get_optim_policies()

    normalize = GroupNormalize(input_mean, input_std)
    normalize_1 = GroupNormalize(input_mean_1, input_std_1)

    dataset = TSNDataSet("train", num_segments=args.num_segments,
                         context=args.context,
                         new_length=data_length,
                         modality=args.modality,
                         image_tmpl="img_{:05d}.jpg" if args.modality in [
                             "RGB"] else args.flow_prefix + "{}_{:05d}.jpg",
                         transform=torchvision.transforms.Compose([
                             GroupScale((224, 224)),
                             Stack(roll=args.arch == 'BNInception'),
                             ToTorchFormatTensor(div=args.arch != 'BNInception'),
                             normalize,
                         ]),
                         transform_1=torchvision.transforms.Compose([
                             GroupScale((224, 224)),
                             Stack(roll=args.arch == 'BNInception'),
                             ToTorchFormatTensor(div=args.arch != 'BNInception'),
                             normalize_1,
                         ]))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("val", num_segments=args.num_segments,
                   context=args.context,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB"] else args.flow_prefix + "{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale((int(224), int(224))),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]),
                   transform_1=torchvision.transforms.Compose([
                       GroupScale((224, 224)),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize_1,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    logger = config.get_logger('train')
    logger.info(model)
    logger.info(vars(args))
    # get function handles of loss and metrics
    criterion_categorical = getattr(module_loss, config['loss'])
    criterion_continuous = getattr(module_loss, config['loss_continuous'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    metrics_continuous = [getattr(module_metric, met) for met in config['metrics_continuous']]

    optimizer_main = torch.optim.SGD(policies,
                                     args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_main)

    for param_group in optimizer_main.param_groups:
        print(param_group['lr'])

    trainer = Trainer(model, criterion_categorical, criterion_continuous, metrics, metrics_continuous, optimizer_main,
                      config=config,
                      data_loader=train_loader,
                      valid_data_loader=val_loader,
                      lr_scheduler=lr_scheduler, embed=args.embed)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow', 'RGBDiff', 'depth'])
    parser.add_argument('--modality_1', type=str, default='Flow', choices=['RGB', 'Flow', 'RGBDiff', 'depth'])

    # ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--num_segments', type=int, default=3)
    parser.add_argument('--consensus_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
    parser.add_argument('--num_factors', type=int, default=3)
    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')

    # ========================= Learning Configs ==========================
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
    parser.add_argument('--context', default=False, action="store_true")
    parser.add_argument('--embed', default=False, action="store_true")

    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--flow_prefix', default="", type=str)
    parser.add_argument('--opn', type=str, default='mul')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--st_gcn_dim', default=256, type=int)
    parser.add_argument('--center_dim', default=512, type=int)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp_name'], type=str, target='name'),
    ]
    config = ConfigParser.from_args(parser, options)
    print(config)

    args = parser.parse_args()
    main(args, config)
