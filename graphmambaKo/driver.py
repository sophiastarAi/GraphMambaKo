import argparse
import sys

import numpy
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from read_dataset import data_from_name
from model import *
from model_gcn import *
import train_val
from tools import *
import scipy as sp
import os
import util
import time

# ==============================================================================
# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--model', type=str, default='GraphMambaKo', metavar='N', help='model')
parser.add_argument('--alpha', type=int, default='1', help='model width')
parser.add_argument('--dataset', type=str, default='data/', metavar='N', help='dataset')
parser.add_argument('--lr', type=float, default=0.01, metavar='N', help='learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=0, metavar='N', help='weight_decay (default: 1e-5)')
parser.add_argument('--epochs', type=int, default=3, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--batch', type=int, default=256, metavar='N', help='batch size (default: 10000)')
parser.add_argument('--folder', type=str, default='test', help='specify directory to print results to')
parser.add_argument('--lamb', type=float, default='1', help='balance between reconstruction and prediction loss')
parser.add_argument('--nu', type=float, default='1e-1', help='tune backward loss')
parser.add_argument('--eta', type=float, default='1e-1', help='tune consistent loss')
parser.add_argument('--steps', type=int, default='1', help='steps for learning forward dynamics')
parser.add_argument('--steps_back', type=int, default='24', help='steps for learning backwards dynamics')
parser.add_argument('--bottleneck', type=int, default='32', help='size of lower embedding')
parser.add_argument('--lr_update', type=int, nargs='+', default=[100,200, 300,400],
                    help='decrease learning rate at these epochs')
parser.add_argument('--lr_decay', type=float, default='0.04', help='PCL penalty lambda hyperparameter')
parser.add_argument('--gamma', type=float, default='0.99', help='learning rate decay')
parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
parser.add_argument('--init_scale', type=float, default=0.96, help='init scaling')
parser.add_argument('--gradclip', type=float, default=1, help='gradient clipping')
parser.add_argument('--pred_steps', type=int, default='24', help='prediction steps')
parser.add_argument('--seed', type=int, default='999', help='seed value')
parser.add_argument('--gpu', type=str, default='-1', help='gpu')
parser.add_argument('--early_stop', type=int, default='100', help='early stop')
parser.add_argument('--mode_constraint', type=int, default='0', help ='0:do not need mode constraint,1:need mode constraint ')

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)
device = get_device()

# ******************************************************************************
# Create folder to save results
# ******************************************************************************
proj_dir = os.path.dirname(os.path.abspath(__file__))

result_path = os.path.join(proj_dir, args.folder)
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

modelpath = result_path
logger = get_logger(os.path.join(modelpath, 'exp.log'))
print(f'model path:{modelpath}')


index_file = os.path.join(proj_dir, 'adj matrix/edge_index.npy')
attr_file = os.path.join(proj_dir, 'adj matrix/edge_attr.npy')
edge_index = np.load(index_file)
edge_attr = np.load(attr_file)
edge_index = torch.tensor(edge_index,dtype=torch.int64)
edge_attr = torch.tensor(edge_attr,dtype=torch.int64)
edge_index = edge_index.to(device)
edge_attr = edge_attr.to(device)
adj=0

dataloader = util.load_dataset(args.dataset, args.batch, args.batch, args.batch)
# train_loader = dataloader['train_loader'].shuffle()
val_loader = dataloader['val_loader']
test_loader = dataloader['test_loader']
scaler = dataloader['scaler']

m = 121
n = 1

# ==============================================================================
# Model,bottleneck
# ==============================================================================
model = GraphMambaKo(m, n, args.bottleneck, args.steps, args.steps_back,args.steps, 128, 64, device, args.alpha, args.init_scale, args.batch)
model = model.to(device)
print(model)

# ==============================================================================
# Start training
# ==============================================================================
model = train_val.train(model, modelpath, dataloader, logger, scaler,
                        lr=args.lr, gamma = args.gamma, weight_decay=args.wd, lamb=args.lamb, num_epochs=args.epochs,
                        learning_rate_change=args.lr_decay, epoch_update=args.lr_update, early_stop=args.early_stop,
                        adj=adj, edge_index=edge_index, edge_attr=edge_attr,
                        nu=args.nu, eta=args.eta, backward=args.backward, steps=args.steps, steps_back=args.steps_back,
                        gradclip=args.gradclip, mode_cons=args.mode_constraint)
# ******************************************************************************
# Prediction
# ******************************************************************************
criterion = nn.MSELoss().to(device)
steps = 1

model.load_state_dict(torch.load(modelpath + '/model' + '.pkl', map_location=device, weights_only=True))
model = model.to(device)

model.eval()

outputs = []
testy = torch.Tensor(dataloader['y_test']).to(device)
for batch_idx, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
# for batch_idx, data_list in enumerate(test_loader):
    testx = torch.Tensor(x).to(device)
    testy = torch.Tensor(y).to(device)
    with torch.no_grad():

        out, out_back = model(testx,edge_index,edge_attr, mode='forward')

    outputs.append(out[0])

yhat = torch.cat(outputs, dim=0)
yhat = yhat[:testy.size(0), ...]
# print("yhat", yhat.shape)

print("Training finished")
# print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

amae = []
amape = []
armse = []
for i in range(args.pred_steps):
    pred = scaler.inverse_transform(yhat[:, i, :])  #64,`1,121
    real = testy[:, i, :]
    #print("pred",pred.shape)
    #print("real", real.shape)
    metrics = util.metric(pred, real)
    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
    amae.append(metrics[0])
    amape.append(metrics[1])
    armse.append(metrics[2])

log = 'On average over horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
