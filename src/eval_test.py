import csv
import re
import os
import random
from collections import Counter
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
#from preprocess_helpers import count_labels
from embedding_utils import *
from evaluate import *
from evaluate_multi import *
from loss import *
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(description='MIMIC III notes data preparation')
parser.add_argument('--train_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_train_data.pkl')
parser.add_argument('--test_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_test_data.pkl')
parser.add_argument('--model_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/models/test.pth')
parser.add_argument('--attention', type=int, default=0)
parser.add_argument('--cbow', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--embed_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=100)
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
    #torch.backends.cudnn.enabled = False

# data reader
log_path = os.path.join('log', args.exp_name)
if not os.path.exists(log_path):
    os.makedirs(log_path)
else:
    exit("Log path already exists. Enter a new exp_name")

args.model_dir = os.path.join(args.model_dir, args.exp_name)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

tensorboard_logger.configure(log_path)


## MIMIC data code
traindata = pickle.load(open(args.train_path, 'rb'))
testdata = pickle.load(open(args.test_path, 'rb'))
print("Train size:", len(traindata))
print("Test size:", len(valdata))

if args.build_starspace:
    print("Building starspace embeddings. This will take a few minutes...")
    stsp_data = convert_unflat_data_to_starspace_format(traindata)
    write_starspace_format(stsp_data, "stsp_embeddings.txt")
    subprocess.call(['./Starspace/run.sh'])

label_map = {i:_ for _,i in enumerate(get_labels(traindata))}
print(label_map)
print(len(label_map))
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
count_labels(testdata)
num_labels = len(label_map)
print("Number of labels is {}".format(num_labels))
if args.use_starspace:
    # Load starspace embeddings into a dict
    #stsp_embed = load_starspace_embeds("Starspace/stsp_model.tsv", args.embed_dim)
    stsp_embed = load_starspace_embeds(args.embed_path, args.embed_dim)
    print(type(stsp_embed))
    print("Embeddings loaded")
    for i, k in enumerate(stsp_embed):
        print(k, stsp_embed[k])
        if i == 3:
            break
    # Create starspace embedding matrix
    emb_mat = create_starspace_embedding_matrix(stsp_embed,
                                                token2idx,
                                                len(vocabulary),
                                                args.embed_dim)
    print("Embedding matrix created")

one_hot = False
if args.multilabel:
    print("Labels are one hot")
    one_hot = True
trainset = NotesData(traindata, token2idx, UNKNOWN, label_map, one_hot)
testset = NotesData(testdata, token2idx, UNKNOWN, label_map, one_hot)
print("Data Loaded in %.2f mns."%((time.time()-_t)/60))

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers, collate_fn=sent_batch_collate)
val_loader = torch.utils.data.DataLoader(dataset = testset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers, collate_fn=sent_batch_collate)
print("data loader done")

if args.attention:
    print("Using hierachical attention model")
    #model = Ensemble(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
    model = HANModel(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
else:
    if args.cbow:
        print("Using CBOW model")
        model = CBOWSentModel(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
    else:
        print("Using Hierachical model")
        model = WordSentModel(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
print(model)

model.load_state_dict(torch.load(args.model_path))
print("Pretrained model loaded")

# model.apply(xavier_weight_init)
if args.focalloss:
    print("Using focal loss")
    crit = FocalLoss(len(label_map), use_cuda)
else:
    if args.multilabel:
        print("Mutilabel, using BCE loss")
        crit = nn.BCEWithLogitsLoss()
    else:
        print("Using cross entropy loss")
        crit = nn.CrossEntropyLoss()
opti = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
# opti = torch.optim.RMSprop(model.parameters(), lr=args.lr)

if use_cuda:
    model.cuda()
    if args.focalloss == 0:
      crit.cuda()
    if args.attention:
        model.wordattention.context = model.wordattention.context.cuda()
        model.sentattention.context = model.sentattention.context.cuda()

print("Evaluating on training set, multilabel eval")
print("Number of labels is {}".format(num_labels))
train_loss, train_acc, train_f1, train_precision, train_recall = eval_model_multi(model, train_loader, args.batch_size, crit, use_cuda, num_labels, args.batch_size)
print("Evaluating on test set, multilabel eval")
test_loss, test_acc, test_f1, test_precision, test_recall = eval_model_multi(model, test_loader, args.batch_size, crit, use_cuda, num_labels, args.batch_size)
