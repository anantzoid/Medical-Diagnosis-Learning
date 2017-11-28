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
from models import LSTMModel
PADDING = '<PAD>'
num_workers = 1

parser = argparse.ArgumentParser(description='MIMIC III training script')
parser.add_argument('--datafolder', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data', help="folder where data is located")
parser.add_argument('--dataset', type=str, default='10codesL5_UNK_content_4', help="dataset prefix")
parser.add_argument('--model', type=str, default='LSTM', help="Which model to use")
parser.add_argument('--modelfolder', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/models', help="Where to save the models")
parser.add_argument('--experiment', type=str, default='test', help="experiment name")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--batchsize', type=int, default=32, help="Batch size")
parser.add_argument('--embeddim', type=int, default=50, help="Embedding dim")
parser.add_argument('--hiddendim', type=int, default=50, help="Hidden dim")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--cuda', type=int, default=0, help="Batch size")
parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use")
parser.add_argument('--stop', type=int, default=0, help="Whether to remove stopwords")
args = parser.parse_args()
print(args)
print()

train_data = pickle.load(open(os.path.join(args.datafolder, args.dataset + "_train_data.pkl"), 'rb'))
valid_data = pickle.load(open(os.path.join(args.datafolder, args.dataset + "_valid_data.pkl"), 'rb'))
labels = pickle.load(open(os.path.join(args.datafolder, args.dataset + "_labels.pkl"), 'rb'))
print("Training and validation data loaded")
if args.stop:
  print("Removing stopwords...")
  train_data = remove_stopwords(train_data)
  valid_data = remove_stopwords(valid_data)
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
if args.cuda:
    print("Using cuda")
    torch.cuda.device(args.gpu)
    model.cuda()
    crit.cuda()
else:
    print("Using CPU only")
params = list(model.parameters())
opti = torch.optim.Adam(params, lr=args.lr)

def evaluate(model, loader, crit, cuda, bs, num_labels):
    data_size = len(loader)
    avg_length = 0
    correct = 0
    total = 0
    total_loss = 0
    true_pos = np.zeros(num_labels)
    true_neg = np.zeros(num_labels)
    false_pos = np.zeros(num_labels)
    false_neg = np.zeros(num_labels)
    last_pred = None
    last_y = None
    for j, (data, labels) in enumerate(loader):
        x = Variable(data)
        y = Variable(labels)
        if cuda:
            x = x.cuda()
            y = y.cuda()
        if x.size(0) != bs:
            continue
        avg_length += x.data.size(1)
        model.eval()
        out = model(x)
        out = out.double()
        loss = crit(out, y)
        total_loss += loss.data[0]
        predicted = out.data > 0.5
        predicted = predicted.int()
        last_pred = predicted
        y = y.data.int()
        last_y = y
        total += data.size(0)
        correct += (predicted == y).sum()
        predicted = predicted.cpu().numpy()
        y = y.cpu().numpy()
        for k in range(y.shape[1]):
          true_pos[k] += np.sum((predicted[:,k][np.where(y[:,k] == 1)] == y[:,k][np.where(y[:,k] == 1)]))
          true_neg[k] += np.sum((predicted[:,k][np.where(y[:,k] == 0)] == y[:,k][np.where(y[:,k] == 0)]))
          false_pos[k] += np.sum((predicted[:,k][np.where(y[:,k] == 1)] != y[:,k][np.where(y[:,k] == 1)]))
          false_neg[k] += np.sum((predicted[:,k][np.where(y[:,k] == 0)] != y[:,k][np.where(y[:,k] == 0)]))
        if j % 50 == 0:
            print("Processed {} batches".format(j+1))
    micro_precision = np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_pos))
    micro_recall = np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_neg))
    micro_F = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
    print("correct: {}, loss = {}, data_size: {}, prediction egs: {}, out egs: {}".format(
              correct / float(total), total_loss / float(data_size), data_size * bs, last_pred[:5], last_y[:5]))
    print("True pos: {}, True neg: {}, False pos: {}, False neg: {}".format(
              np.sum(true_pos), np.sum(true_neg), np.sum(false_pos), np.sum(false_neg)))
    print("Micro precision: {:4f}, micro recall: {:4f}, micro F1: {:4f}".format(micro_precision, micro_recall, micro_F)) 


for i in range(args.epochs):
    data_size = len(train_loader)
    avg_length = 0
    num_labels = None
    for j, (data, labels) in enumerate(train_loader):
        x = Variable(data)
        y = Variable(labels)
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        # print(x.data.shape, y.data.shape)
        num_labels = y.data.size(1)
        avg_length += x.data.size(1)
        model.train()
        opti.zero_grad()
        model.zero_grad()
        out = model(x)
        out = out.double()
        loss = crit(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        opti.step()
        if j % 50 == 0:
            print("Epoch: {}, Batch: {}/{}, loss: {} average seq length: {}, data size: {}".format(
                    i, j, data_size, loss.data[0], avg_length / float(j + 1), data_size * args.batchsize))
    print("Evaluating model on training data")
    evaluate(model, train_loader, crit, args.cuda, args.batchsize, num_labels)
    print("Evaluating model on validation data")
    evaluate(model, valid_loader, crit, args.cuda, args.batchsize, num_labels)
    if i % 2 == 0:
      torch.save(model.state_dict(), os.path.join(args.modelfolder, args.experiment + '_' + str(i)))
