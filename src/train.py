from collections import Counter
import numpy as np
import pickle
import argparse
import os
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
from build_datasets_utils import *
from evaluate import *
from models import LSTMModel
PADDING = '<PAD>'
num_workers = 1

parser = argparse.ArgumentParser(description='MIMIC III training script')
parser.add_argument('--datafolder', type=str, default='/Users/lauragraesser/Documents/NYU_Courses/medical_data', help="folder where data is located")
parser.add_argument('--dataset', type=str, default='10codesL5_UNK_content_4', help="dataset prefix")
parser.add_argument('--model', type=str, default='LSTM', help="Which model to use")
parser.add_argument('--experiment', type=str, default='test', help="experiment name")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--batchsize', type=int, default=32, help="Batch size")
parser.add_argument('--embeddim', type=int, default=50, help="Embedding dim")
parser.add_argument('--hiddendim', type=int, default=50, help="Hidden dim")
parser.add_argument('--cuda', type=int, default=0, help="Batch size")
args = parser.parse_args()
print(args)
print()

train_data = pickle.load(open(os.path.join(args.datafolder, args.dataset + "_train_data.pkl"), 'rb'))
valid_data = pickle.load(open(os.path.join(args.datafolder, args.dataset + "_valid_data.pkl"), 'rb'))
labels = pickle.load(open(os.path.join(args.datafolder, args.dataset + "_labels.pkl"), 'rb'))
print("Training and validation data loaded")
(word_2_idx, vocab) = build_dictionary(train_data, PADDING)
label_map = build_label_map(labels)
print("Labels: {}\n, label map: {}".format(labels, label_map))
train_dataset = FlatData(train_data, word_2_idx, label_map)
valid_dataset = FlatData(valid_data, word_2_idx, label_map)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batchsize,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           collate_fn=flat_batch_collate)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=args.batchsize,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           collate_fn=flat_batch_collate)

# Init models, opt and criterion
if args.model is 'LSTM':
    model = LSTMModel(len(vocab), args.embeddim, args.hiddendim, label_map, args.batchsize, args.cuda)
else:
    # To fix with more models
    model = LSTMModel(len(vocab), args.embeddim, args.hiddendim, label_map, args.batchsize, args.cuda)
crit = nn.BCEWithLogitsLoss()
params = list(model.parameters())
opti = torch.optim.Adam(params)
if args.cuda:
    print("Using cuda")
    model.cuda()
    opti.cuda()
else:
    print("Using CPU only")

for i in range(args.epochs):
    data_size = len(train_loader)
    for j, (data, labels) in enumerate(train_loader):
        x = Variable(data)
        y = Variable(labels)
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        # print(x.data.shape, y.data.shape)
        model.train()
        out = model(x)
        out = out.double()
        loss = crit(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        opti.step()
        if j % 1 == 0:
            print("Epoch: {}, Batch: {}/{}, loss: {}".format(i, j, data_size, loss))
    evaluate(model, valid_loader, args.batchsize, crit, args.cuda)
