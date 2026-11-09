"""Train PoSafeNet (Hard or Mixture) on the 2D robot navigation dataset.

Examples
--------
    python train_posafenet.py --model hard    --exp 0
    python train_posafenet.py --model mixture --exp 0

Checkpoints are written every epoch to
    <log-root>/posafenet_<model>/posafenet_<model><exp:02d>/model_posa<epoch:02d>.pth
which is the layout expected by eval_posafenet.py.
"""
import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

import models
from my_classes import Dataset_list as Dataset

plt.style.use('bmh')

_THIS = Path(__file__).resolve()

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", choices=["hard", "mixture"], required=True)
parser.add_argument("--exp", type=int, default=0, help="experiment index used in the output folder name")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--heads", type=int, default=6)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-workers", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=None,
                    help="optional torch/numpy seed (the original runs did not fix a seed)")
parser.add_argument("--data-dir", type=Path, default=_THIS.parent / "data_v2" / "data")
parser.add_argument("--log-root", type=Path, default=_THIS.parent / "log")
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

exp = args.exp
run_name = f"posafenet_{args.model}"
run_dir = args.log_root / run_name / f"{run_name}{exp:02d}"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True


# Datasets
DATA_DIR = args.data_dir
train_data = sio.loadmat(str(DATA_DIR / "dataM_train.mat"))["data"]
valid_data = sio.loadmat(str(DATA_DIR / "dataM_valid.mat"))["data"]
test_data  = sio.loadmat(str(DATA_DIR / "dataM_test.mat"))["data"]


train0 = np.double(train_data[:,2:7])  # px, py, theta, v, dst_y, //dst_x is fixed
train_labels = np.reshape(np.double(train_data[:,7:9]), (len(train_data),2)) #theta_derivative, acc
valid0 = np.double(valid_data[:,2:7])
valid_labels = np.reshape(np.double(valid_data[:,7:9]), (len(valid_data),2))
test0 = np.double(test_data[:,2:7])
test_labels = np.reshape(np.double(test_data[:,7:9]), (len(test_data),2))

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)

train0 = (train0 - mean)/std
valid0 = (valid0 - mean)/std
test0 = (test0 - mean)/std


# Parameters
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers}

# Generators
training_set = Dataset(train0, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

valid_set = Dataset(valid0, valid_labels)
valid_dataloader = torch.utils.data.DataLoader(valid_set, **params)


# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2
if args.model == "hard":
    model = models.PoSafeNet_Hard(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads=args.heads).to(device)
else:
    model = models.PoSafeNet_Mixture(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads=args.heads).to(device)
print(model)


# Initialize the optimizer.
learning_rate = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def train(dataloader, model, loss_fn, optimizer, losses, itr):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X, 1)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:  #25
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def test(dataloader, model, loss_fn, losses, itr):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X, 1)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    losses.append(test_loss)
    print(f"Test avg loss: {test_loss:>8f} \n")
    return losses


epochs = args.epochs
train_losses, test_losses = [], []
start_time = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = train(train_dataloader, model, loss_fn, optimizer, train_losses, t)
    test_losses = test(valid_dataloader, model, loss_fn, test_losses, t)
    makedirs(run_dir)
    ckpt = run_dir / f"model_posa{t:02d}.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved PyTorch Model State to {ckpt}")
print("Training Done!")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time/3600} hours")


model.eval()

# Open-loop prediction on the test split (for the plots below)
X_np = np.asarray(test0)          # (N, nFeatures)
Y_np = np.asarray(test_labels)    # (N, 2)

tr = (np.arange(len(X_np)) * 0.1).tolist()

with torch.no_grad():
    X = torch.from_numpy(X_np).to(device=device, dtype=torch.float64)  # (N, nFeatures)

    chunk = 4096
    U_list = []

    for s in range(0, X.shape[0], chunk):
        xb = X[s:s+chunk]  # (B, nFeatures)
        ub = model(xb, 0)

        if ub.ndim == 1:
            ub = ub.view(1, -1)
        if ub.shape[-1] != 2:
            raise RuntimeError(f"Model output must have last dim=2, got {ub.shape}")

        U_list.append(ub)

    U = torch.cat(U_list, dim=0)  # (N,2)

    U_cpu = U.detach().cpu().numpy()   # (N,2)
    ctrl1 = U_cpu[:, 0].tolist()
    ctrl2 = U_cpu[:, 1].tolist()

ctrl1_real = Y_np[:, 0].tolist()
ctrl2_real = Y_np[:, 1].tolist()

print("Test done!")


plt.figure(1)
plt.plot(tr, ctrl1_real, color = 'red', label = 'actual(optimal)')
plt.plot(tr, ctrl1, color = 'blue', label = 'implemented')
plt.legend()
plt.ylabel('Angular speed (control)')
plt.xlabel('time')
plt.savefig(run_dir / 'train_test_u1')

plt.figure(2)
plt.plot(tr, ctrl2_real, color = 'red', label = 'actual(optimal)')
plt.plot(tr, ctrl2, color = 'blue', label = 'implemented')
plt.legend()
plt.ylabel('Acceleration (control)')
plt.xlabel('time')
plt.savefig(run_dir / 'train_test_u2')

plt.figure(3)
plt.plot(train_losses, color = 'green', label = 'train')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('time')
plt.ylim(ymin=0.)
plt.savefig(run_dir / 'train_loss')
print("end")
