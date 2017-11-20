import os
import pickle
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import re
import time
import random
use_cuda = torch.cuda.is_available()
from util_icu_train import *

if use_cuda:
    torch.cuda.set_device(0)

base_path = '/misc/vlgscratch2/LecunGroup/anant/nlp'
batch_size = 16
num_workers = 2
emb_size = 100
hidden_dim = 300
random.seed(10)
lr = 0.01

traindata, valdata = read_data_dump_v3(os.path.join(base_path, 'notes_dump_cleaned.pkl'))
token2idx = {i:_ for _,i in enumerate(list(set(gett2i(traindata) + gett2i(valdata))))}
for i,item in enumerate(traindata):
    traindata[i]['notes'] = [token2idx[_] for _ in item['notes']]

label_map = get_labels(os.path.join(base_path, 'labels.txt'))
model = LSTMModel(emb_size, hidden_dim, len(label_map.keys()), batch_size, len(token2idx.keys()), use_cuda)

opti = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
crit = nn.CrossEntropyLoss()

if use_cuda:
    model.cuda()
    crit.cuda()

def collate_func(batch):
    data_list = []
    label_list = []

    max_length = np.max([len(_['notes']) for _ in batch])
    # padding
    for datum in batch:
        data_list.append([0]*(max_length-len(datum['notes'])) + datum['notes'])
        label_list.append(datum['labels']['icd'][0])
    return (data_list, label_list)

train_loader = torch.utils.data.DataLoader(dataset=NoteDataloader(traindata), 
                batch_size=batch_size, shuffle=True, collate_fn=collate_func, num_workers=num_workers)

avg_loss = []
for nup in range(100):
    for step, i in enumerate(train_loader):
        if len(i[0]) != batch_size:
            continue

        model.zero_grad()
        x = Variable(torch.LongTensor([i[0]])).view(batch_size, -1)
        y = Variable(torch.LongTensor([label_map[_] for _ in i[1]]))
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        hidden = model.init_hidden()
        x = model(x, hidden)
        loss = crit(x, y)
        loss.backward()
        opti.step()
        avg_loss.append(loss.data[0])
        if step%50==0:
            _, predicted = torch.max(x.data, 1)
            trainacc = (predicted == y.data).sum()/float(batch_size)
            print "e: %d, s: %d, loss: %.4f, avg_acc: %.3f"%(nup, step, np.mean(avg_loss), trainacc)            
            
    if nup==30:
        lr = 0.001
        opti = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        print "new lr:", lr
    
    if nup==60:
        lr = 0.0001
        opti = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        print "new lr:", lr
    
    model.save_state_dict('v2.pth')
