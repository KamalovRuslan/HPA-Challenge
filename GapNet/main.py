from loader import HumanProteinDataset
from model import GapNet, f1_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import argparse

from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from torch.autograd import Variable

from torch.nn import MultiLabelMarginLoss

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np

PATH = './'
TRAIN = '../data/train/'
TEST = '../data/test/'
LABELS = '../data/train.csv'
SAMPLE = '../data/sample_submission.csv'

parser = argparse.ArgumentParser(description='PyTroch Human Protein Atlas Chalenge GapNet Model')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--bsize', type=int, default=16, metavar='BS',
                    help='batch size (default: 16)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print("Build model...")
model = GapNet()
criterion = MultiLabelMarginLoss()
if args.cuda:
    model.cuda()
    criterion.cuda()
    print("Model on GPU!")

print(args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print("Build Dataset...")
train_dataset = HumanProteinDataset(LABELS, TRAIN)
train_loader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=False,
                                         pin_memory=True, num_workers=4)

def train(epoch):
    total_loss = 0.
    model.train()
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        img, label = batch['image'], batch['label']
        img, label = Variable(img, requires_grad=True).cuda(), Variable(label.long()).cuda()
        optimizer.zero_grad()
        
        probs = model(img)
        loss = criterion(probs, label)
        # loss = f1_loss(label, probs)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    # total_loss /= len(train_loader)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, total_loss))

def test(thresh):
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        img, label = batch['image'], batch['label']
        img, label = Variable(img, volatile=True).cuda(), Variable(label, volatile=True).cuda()
        probs = model(img)
        if i == 0:
            y_pred = probs.cpu().data > thresh
            y_pred = y_pred.numpy()
            y_true = label.cpu().data.numpy()
        else:
            probs = probs.cpu().data > thresh
            y_pred = np.vstack((y_pred, probs.numpy()))
            y_true = np.vstack((y_true, label.cpu().data.numpy()))

    score = f1_score(y_true, y_pred, average='macro')
    print("Test F1-score : {:.5f}".format(score))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(0.5)
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join('models', 'model.pth'))
