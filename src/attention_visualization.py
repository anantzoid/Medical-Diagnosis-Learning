import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import math
import tensorboard_logger
import pickle
import time
import argparse
import subprocess

from attention_databuilder import *
from attention_models import *
from embedding_utils import *
from evaluate import *
from evaluate_multi import *
from loss import *
from sklearn.metrics import f1_score, precision_score, recall_score
import json

parser = argparse.ArgumentParser(description='MIMIC III notes data preparation')
parser.add_argument('--load_path', type=str, default='10codesL5_UNK_content_4_top1.pth')
parser.add_argument('--exp_name', type=str, default='run')
parser.add_argument('--train_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_train_data.pkl')
parser.add_argument('--val_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_valid_data.pkl')
parser.add_argument('--model_dir', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/models/')
parser.add_argument('--attention', type=int, default=0)
parser.add_argument('--cbow', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--embed_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--focalloss', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_decay_rate', type=float, default=0.9)
parser.add_argument('--lr_decay_epoch', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--vocab_threshold', type=int, default=20)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--build_starspace', type=int, default=0)
parser.add_argument('--use_starspace', type=int, default=1)
parser.add_argument('--multilabel', type=int, default=0)
parser.add_argument('--embed_path', type=str, default="Starspace/stsp_model.tsv",
                    help='Where are the initialized embeddings?')
args = parser.parse_args()
print(args)

PADDING = "<PAD>"
UNKNOWN = "UNK"
_t = time.time()

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu_id)

## MIMIC data code
traindata = pickle.load(open(args.train_path, 'rb'))
#valdata = pickle.load(open(args.val_path, 'rb'))
print("Train size:", len(traindata))
#print("Valid size:", len(valdata))
# traindata = chf_data(traindata)
# valdata = chf_data(valdata)
# print("CHF Train size:", len(traindata))
# print("CHF Valid size:", len(valdata))

label_map = {i:_ for _,i in enumerate(get_labels(traindata))}
reverse_label_map = {j:i for i,j in label_map.items()}
vocabulary, token2idx  = build_vocab(traindata, PADDING, UNKNOWN, args.vocab_threshold)
print("Vocab size:", len(vocabulary))
print(vocabulary[:20])
print("====================== Data examples =======================")
for i in range(10):
    print(traindata[i])
    print("=============================================================")
print("=============================================================")
print(list(token2idx.keys())[:20])
print(label_map)
print(list(token2idx.values())[:20])
print("Label mix training data")
count_labels(traindata)
print("Label mix valid data")
#count_labels(valdata)

one_hot = False
if args.multilabel:
    print("Labels are one hot")
    one_hot = True
trainset = NotesData(traindata, token2idx, UNKNOWN, label_map, one_hot)
print("Data Loaded in %.2f mns."%((time.time()-_t)/60))

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers, collate_fn=sent_batch_collate)
print("Using hierachical attention model")
print("model params:", args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
model = HANModel(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)

load_path = '10codesL5_UNK_content_4_top1.pth'
model.load_state_dict(torch.load(args.load_path))

if use_cuda:
    model.cuda()
    if args.attention:
        model.wordattention.context = model.wordattention.context.cuda()
        model.sentattention.context = model.sentattention.context.cuda()

print("Starting training...")
step = 0
eg = True
#train_loss_mean = []
alljson = []
for enum, batch in enumerate(train_loader):
    if batch[0].size(0) != args.batch_size:
        continue
    word_hidden = model.word_rnn.init_hidden(batch[0].size(0) * batch[0].size(1))
    sent_hidden = model.sent_rnn.init_hidden()
    if use_cuda:
        word_hidden, sent_hidden = word_hidden.cuda(), sent_hidden.cuda()

    model.eval()
    batch_x = Variable(batch[0])
    batch_y = Variable(batch[1])
    length_x = batch[2].type(torch.FloatTensor).unsqueeze(1)
    length_x = Variable(length_x)

    if use_cuda:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        length_x = length_x.cuda()

    pred_prob, word_w, sent_w = model(batch_x, word_hidden, sent_hidden, length_x)
    sent_w = sent_w.view(-1).data.cpu().numpy()
    word_w = word_w.squeeze(2).data.cpu().numpy()

    attention_weights = []
    for _,(i,j) in enumerate(zip(sent_w, batch[3][0])):
        print(i, "==>", "--".join([str(k)+" "+word for k, word in zip(word_w[_,:],j)]))
        attention_weights.append({"sent_w": float(i), "word_w": [[float(k), word] for k, word in zip(word_w[_,:],j)]})
    print("label:", reverse_label_map[batch[1][0]])
    alljson.append({"label": reverse_label_map[batch[1][0]], "data": attention_weights})
    if enum > 10:
        break

with open(args.load_path+"_vis.json", 'w') as f:
    json.dump({"data": alljson}, f)
        
