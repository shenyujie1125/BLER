import argparse
import time
import csv
import model.metric
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import torchvision
from dataset import *
from transforms import *
from model.models import *
import openpyxl
from model.ops import ConsensusModule
import pandas as pd

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")

parser.add_argument('--mode', type=str, default="test")
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'], default="RGB")
parser.add_argument('--modality_1', type=str, choices=['RGB', 'Flow', 'RGBDiff'], default="Flow")
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--context', default=False, action="store_true")
parser.add_argument('--categorical', default=True, action="store_true")
parser.add_argument('--continuous', default=True, action="store_true")
parser.add_argument('--embed', default=True, action='store_false', help='help message here')
parser.add_argument('--num_factors', type=int, default=3)
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--st_gcn_dim', type=int, default=256)
parser.add_argument('--center_dim', type=int, default=512)
parser.add_argument('--opn', type=str, default='mul')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

args = parser.parse_args()
if args.modality == 'RGB':
    data_length = 1
if args.modality_1 == 'Flow':
    data_length_1 = 5
# args.weights = model
args.test_crops = 1
args.test_segments = 3
print(args)

net = TSN(26, 1, modality=args.modality, modality_1=args.modality_1,
          base_model=args.arch, new_length=data_length, new_length_1=data_length_1,
          consensus_type=args.crop_fusion_type, embed=args.embed, context=args.context,
          dropout=args.dropout, args=args, partial_bn=not args.no_partialbn)
# 1表示num_segment

features_blobs = []
checkpoint = torch.load(args.weights)
# temp = list(checkpoint['state_dict'].items())
# temp1 = list(net.state_dict().items())
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
a = net.load_state_dict(base_dict, strict=True)
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale((224, 224)),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(224, 224)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
    TSNDataSet("{}".format(args.mode), num_segments=args.test_segments, context=args.context,
               new_length=1 if args.modality == "RGB" else 5,
               modality=args.modality,
               image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix + "{}_{:05d}.jpg",
               test_mode=True,
               transform=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=args.arch == 'BNInception'),
                   ToTorchFormatTensor(div=args.arch != 'BNInception'),
                   GroupNormalize(net.input_mean, net.input_std),
               ]),
               transform_1=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=args.arch == 'BNInception'),
                   ToTorchFormatTensor(div=args.arch != 'BNInception'),
                   GroupNormalize(net.input_mean_1, net.input_std_1),
               ])
               ),
    batch_size=1, shuffle=False,
    num_workers=args.workers * 2, pin_memory=True)

devices = [1]
net = torch.nn.DataParallel(net.cuda(), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)
total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, stgcndata, data, data1, label, label_cont = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality " + args.modality)
    if args.modality_1 == 'Flow':
        length_1 = 10
    stgcndata = stgcndata.expand(args.test_segments, -1, -1, -1, -1)
    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)), volatile=True)
    input_var1 = torch.autograd.Variable(data1.view(-1, length_1, data.size(2), data.size(3)), volatile=True)
    input_var2 = torch.autograd.Variable(stgcndata, volatile=True)

    out, _ = net(input_var, input_var1, input_var2)
    rst = torch.sigmoid(out['categorical']).data.cpu().numpy().copy()
    rst_cont = torch.sigmoid(out['continuous']).data.cpu().numpy().copy()

    return i, rst.reshape((num_crop, args.test_segments, 26)).mean(axis=0).reshape(
        (args.test_segments, 1, 26)
    ), rst_cont.reshape((num_crop, args.test_segments, 3)).mean(axis=0).reshape(
        (args.test_segments, 1, 3)
    )


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, batch in data_gen:
    stgcn_data, data, data1, embeddings = batch
    rst = eval_video((i, stgcn_data, data, data1, None, None))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1, total_num, float(cnt_time) / (i + 1)))

video_pred = np.squeeze(np.array([np.mean(x[0], axis=0) for x in output]))
video_pred_cont = np.squeeze(np.array([np.mean(x[1], axis=0) for x in output]))


if args.mode == 'val':
    cols_continue = range(30, 33)
    df_continue = pd.read_csv('/home/a/Documents/BOLD_public/annotations/val.csv', usecols=cols_continue, header=None)
    val_continue_target = np.array(df_continue.values.tolist())/10.0

    cols_class = range(4, 30)
    df_class = pd.read_csv('/home/a/Documents/BOLD_public/annotations/val.csv', usecols=cols_class, header=None)
    val_class_target = np.array(df_class.values.tolist())

    val_class_target[val_class_target >= 0.5] = 1  # threshold to get binary labels
    val_class_target[val_class_target < 0.5] = 0
    ap = model.metric.average_precision(video_pred, val_class_target)
    roc_auc = model.metric.roc_auc(video_pred, val_class_target)

    mse = model.metric.mean_squared_error(video_pred_cont, val_continue_target)
    r2 = model.metric.r2(video_pred_cont, val_continue_target)

    ers = model.metric.ERS(np.mean(r2), np.mean(ap), np.mean(roc_auc))
    print('ap: {} roc_auc:{} mse:{} r2:{} ERS:{}'.format(np.mean(ap), np.mean(roc_auc), np.mean(mse), np.mean(r2), ers))

else:
    concatenated = np.concatenate((video_pred_cont, video_pred), axis=1)
    with open('output_{}_{}.csv'.format(args.weights.split('/')[-3], args.weights.split('/')[-2]), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(concatenated)
