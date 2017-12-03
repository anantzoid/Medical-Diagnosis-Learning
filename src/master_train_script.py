import csv
import re
import os
import random
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import math
import tensorboard_logger
import pickle
import time
import argparse

from attention_databuilder import *
from attention_models import *


parser = argparse.ArgumentParser(description='MIMIC III notes data preparation')
parser.add_argument('--exp_name', type=str, default='run')
parser.add_argument('--train_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/50codesL5_UNK_content_4_top1_train_data.pkl')
parser.add_argument('--val_path', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/50codesL5_UNK_content_4_top1_valid_data.pkl')
parser.add_argument('--model_file', type=str, default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/models/testv1.pth')
parser.add_argument('--attention', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--embed_dim', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_decay_rate', type=float, default=0.9)
parser.add_argument('--lr_decay_epoch', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--log_interval', type=int, default=25)
parser.add_argument('--vocab_threshold', type=int, default=50)
parser.add_argument('--gpu_id', type=int, default=1)
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
tensorboard_logger.configure(log_path)


## MIMIC data code
traindata = pickle.load(open(args.train_path, 'r'))
valdata = pickle.load(open(args.val_path, 'r'))
print("Train size:", len(traindata))
print("Valid size:", len(valdata))
traindata = chf_data(traindata)
valdata = chf_data(valdata)
print("CHF Train size:", len(traindata))
print("CHF Valid size:", len(valdata))

# Build starspace embeddings
from embedding_utils import *
stsp_data = convert_unflat_data_to_starspace_format(traindata)
write_starspace_format(stsp_data, "stsp_embeddings.txt")
exit()

# Printing examples
for i in range(10):
    print(traindata[i])
exit()

label_map = {i:_ for _,i in enumerate(get_labels(traindata))}
vocabulary, token2idx  = build_vocab(traindata, PADDING, UNKNOWN, args.vocab_threshold)
print("Vocab size:", len(vocabulary))
print(vocabulary[:20])
print(traindata[:10])
print(list(token2idx.keys())[:20])
print(label_map)
print(list(token2idx.values())[:20])

trainset = NotesData(traindata, token2idx, UNKNOWN, label_map)
valset = NotesData(valdata, token2idx, UNKNOWN, label_map)
print("Data Loaded in %.2f mns."%((time.time()-_t)/60))

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers, collate_fn=sent_batch_collate)
val_loader = torch.utils.data.DataLoader(dataset = valset, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=args.num_workers, collate_fn=sent_batch_collate)
print("data loader done")

if args.attention:
    model = Ensemble(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
else:
    model = WordSentModel(args.embed_dim, len(vocabulary), args.hidden_dim, args.batch_size, label_map)
print(model)

# model.apply(xavier_weight_init)
crit = nn.CrossEntropyLoss()
# crit = nn.BCEWithLogitsLoss()
opti = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
# opti = torch.optim.RMSprop(model.parameters(), lr=args.lr)

if use_cuda:
    model.cuda()
    crit.cuda()
    if args.attention:
        model.wordattention.context = model.wordattention.context.cuda()
        model.sentattention.context = model.sentattention.context.cuda()

print("Starting training...")
step = 0
train_loss_mean = []
for n_e in range(args.num_epochs):

    for batch in train_loader:
        if batch[0].size(0) != args.batch_size:
            continue
        # print("Num padded sentences: {}, Num padded words per sentence: {}".format(
            # batch[0].size(1), batch[0].size(2)))
        word_hidden = model.word_rnn.init_hidden(batch[0].size(0) * batch[0].size(1))
        sent_hidden = model.sent_rnn.init_hidden()
        if use_cuda:
            word_hidden, sent_hidden = word_hidden.cuda(), sent_hidden.cuda()

        #model.zero_grad()
        opti.zero_grad()
        batch_x = Variable(batch[0])
        batch_y = Variable(batch[1])

        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        pred_prob = model(batch_x, word_hidden, sent_hidden)
        loss = crit(pred_prob, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        opti.step()

        train_loss_mean.append(loss.data[0])

        if step % args.log_interval ==0:
            val_loss_mean = 0
            correct = 0
            for val_batch in val_loader:
                if batch[0].size(0) != args.batch_size:
                    continue

                word_hidden = model.word_rnn.init_hidden(batch[0].size(0) * batch[0].size(1), True)
                sent_hidden = model.sent_rnn.init_hidden(True)
                if use_cuda:
                    word_hidden, sent_hidden = word_hidden.cuda(), sent_hidden.cuda()


                batch_x, val_batch_y = Variable(batch[0], volatile=True), Variable(batch[1])
                if use_cuda:
                    batch_x, val_batch_y = batch_x.cuda(), val_batch_y.cuda()

                outputs = model(batch_x, word_hidden, sent_hidden)
                val_loss = crit(outputs, val_batch_y)
                val_loss_mean += val_loss.data[0]

                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(val_batch_y.data).cpu().sum()

            train_loss_mean = np.mean(train_loss_mean)
            print(correct, len(val_loader.dataset))
            correct /= float(len(val_loader.dataset))
            val_loss_mean /= float(len(val_loader.dataset))
            print("Epoch: %d, Step: %d, Train Loss: %.2f, Val Loss: %.2f, Val acc: %.3f"%(n_e, step, train_loss_mean, val_loss_mean, correct))

            param1, grad1 = calc_grad_norm(model.parameters(), 1)
            param2, grad2 = calc_grad_norm(model.parameters(), 2)
            print("Param Norm1: %.2f, grad Norm1: %.2f, Param Norm12: %.2f, grad Norm2: %.2f"%(param1, grad1, param2, grad2))

            tensorboard_logger.log_value('train_loss', train_loss_mean, step)
            tensorboard_logger.log_value('val_loss', val_loss_mean, step)
            tensorboard_logger.log_value('val_acc', correct, step)
            tensorboard_logger.log_value('param norm1', param1, step)
            tensorboard_logger.log_value('grad norm1', grad1, step)
            tensorboard_logger.log_value('param norm2', param2, step)
            tensorboard_logger.log_value('grad norm2', grad2, step)
            train_loss_mean = []

        step += 1
    if n_e % args.lr_decay_epoch == 0:
        args.lr *= args.lr_decay_rate
        print("LR changed to", args.lr)
        opti = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(predicted[:20])
    print(val_batch_y[:20])

    torch.save(model.state_dict(), args.model_file)
