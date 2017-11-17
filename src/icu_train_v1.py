# coding: utf-8
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
base_path = '/media/disk3/disk3'
use_cuda = False#torch.cuda.is_available()

batch_size = 1
num_workers = 1
hidden_dim = 400


# In[42]:

def read_data_dump(data_path, token2idx):
    with open(data_path, 'r') as f:
        rawdata = pickle.load(f)['data']

    for i, key in enumerate(rawdata):
        rawdata[i]['notes'] = [[token2idx.get(token, token2idx['unknown']) for token in note.split(' ')] for note in key['notes']]
    random.shuffle(rawdata)
    margin = int(len(rawdata)*0.8)
    traindata = rawdata[:margin]
    valdata = rawdata[margin:]
    return (traindata, valdata)

def read_embeddings(vec_path):    
    vecs, tokens = [], []
    with open(vec_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for i,row in enumerate(tsvreader):
            tokens.append(row[0])
            vecs.append(np.array([float(_) for _ in row[1:]]).reshape(1, -1))
    vecs = np.concatenate(vecs)
    token2idx = {token:_ for _, token in enumerate(tokens)}
    return (vecs, token2idx)

#NOTE to be used later when icd9 labels are too large and
# needs to be mapped to ccs
def get_labels_ccs(ccs_path):
    with open(ccs_path, 'rb') as csvf:
        ccs_mapping = {}
        reader = csv.reader(csvf, delimiter=',')
        _ = next(reader)
        for i, row in enumerate(reader):
            ccs_mapping[row[0]] = row[1]
    ccs2idx = {code: i for i, code in enumerate(list(set([ccs_maping[k] for k in ccs_maping.keys()])))}
    icd2idx = {icd: ccs2idx[ccs_maping[icd]] for icd in ccs_maping.keys()}
    return icd2idx

def get_labels(labels_path):
    f = open(labels_path, 'r')
    labels = []
    for line in f.readlines():
        labels.append(line.replace('\n', ''))
    labels = {i:_ for _,i in enumerate(labels)}
    return labels


class Dataloader(Dataset):
    def __init__(self, data):
        super(Dataloader, self).__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
def padding_collation(batch):
    batch_x, batch_y = [], []
    max_seq_len = np.max([len(key['notes']) for key in batch])
    
    for key in range(len(batch)):
        x = batch[key]['notes']
        note_max_seq_len = np.max([len(note) for note in x])
        #x= [[1759 for i in range(note_max_seq_len-len(note))]+note for note in x]

        # NOTE hardcoded value of 'pad'. Automate this!! Make padding 0 next time
        x = [[1759] for i in range(max_seq_len-len(x))] + x
        #x = [np.array(_) for _ in x]
        y =  batch[key]['labels']['icd'][0]
        batch_x.append(x)
        batch_y.append(y)
    return (batch_x, batch_y)


class LSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, batch_size, pretrained):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        vocab_size = pretrained.shape[0]
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(pretrained).float())

        #Attention!!!
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.lin = nn.Linear(hidden_dim, num_labels)

    def init_hidden(self):
        hidden1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        if use_cuda:
            return (hidden1.cuda(), hidden2.cuda())
        else:
            return (hidden1, hidden2)
    def forward(self, x, hidden):
        # seqlen x batch x emb_dim

        # batch x seq_len x list of idx  -> batch x seq_len x embed_size
        avg_embeds = []
        for b in range(batch_size):
            avg_embeds.append(torch.mean(embed(x[b, :, :]), dim=1))
        x = torch.cat(avg_embeds, dim=0)
        x = torch.transpose(x, 1, 0)
        x, _hidden  = self.lstm(x, hidden)
        x = x[-1, :, :].view(self.batch_size, -1)
        x = self.lin(x)
        return x


# In[144]:

print "Reading data..."
_time = time.time()
pretrained, token2idx = read_embeddings(os.path.join(base_path, 'ri-3gram-400-tsv/filtered_embeddings.tsv'))
traindata, valdata = read_data_dump(os.path.join(base_path, 'notes_dump_cleaned.pkl'), token2idx)
label_map = get_labels(os.path.join(base_path, 'icd9_ccs.csv'))
print "Preprocessing done in %.2f"%((time.time()-_time)/60)


model = LSTMModel(400, hidden_dim, len(label_map.keys()), batch_size, pretrained)

opti = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
model.embed.weight.requires_grad = False
crit = nn.CrossEntropyLoss()

if use_cuda:
    model.cuda()
    crit.cuda()

train_loader = torch.utils.data.DataLoader(dataset=Dataloader(traindata), batch_size=batch_size, shuffle=True, 
                                                           num_workers=num_workers)
#val_loader = torch.utils.data.DataLoader(dataset=Dataloader(valdata), batch_size=batch_size, shuffle=False) 

# In[147]:

def evaluate(model, loader, batch_size, label_map):
    correct = 0
    total = 0

    for i in loader:
        if len(i[0]) != batch_size:
            continue
        seq_len = len(i[0][0])
        batch_x = np.ndarray((batch_size, seq_len, 400))
        for bs, adm in enumerate(i[0]):
            for idx, note in enumerate(adm):
                note_vec = []
                for token in note:
                    note_vec.append(pretrained.get(re.sub(r'[^\w\s]','', token), pretrained['unknown']))
                
                batch_x[bs, idx, :] = np.mean(note_vec)  
                
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(np.array([label_map[_] for _ in i[1]])).long().view(batch_size)
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
        x = Variable(batch_x)
        hidden = model.init_hidden()
        x = model(x, hidden)
        _, predicted = torch.max(x.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum()
        break
    return correct / float(total)


# In[ ]:

step = 0
num_epochs = 100
step_log = []
loss_log = []
val_acc_log = []
train_acc_log = []
train_acc = []
torch.manual_seed(1)
for num_ep in range(num_epochs):
    for i in train_loader:
        if len(i[0]) != batch_size:
            continue
        model.zero_grad()
        for _ in i[0][0]:
            print len(_)
        print np.array(i[0][0]).shape 
        exit()
        seq_len = len(i[0][0])
        #batch_x = np.ndarray((batch_size, seq_len, 400))
        batch_x = np.array(i[0])
        for bs, adm in enumerate(i[0]):
            for idx, note in enumerate(adm):
                note_vec = []
                for token in note:
                    note_vec.append(pretrained.get(re.sub(r'[^\w\s]','', token), pretrained['unknown']))
                
                batch_x[bs, idx, :] = np.mean(note_vec)  
                # batch_x[bs, idx, :] = np.mean([pretrained.get(re.sub(r'[^\w\s]','', token), pretrained['unknown']) for token in note], axis=0)
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(np.array([label_map[_] for _ in i[1]])).long().view(batch_size)
        if use_cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        x = Variable(batch_x)
        hidden = model.init_hidden()
        x = model(x, hidden)

        loss = crit(x, Variable(batch_y))
        loss.backward()
        opti.step()

        _, predicted = torch.max(x.data, 1)
        total = batch_y.size(0)
        correct = (predicted == batch_y).sum()
        train_acc.append(correct / float(total))

        if step % 100 == 0:
            train_acc = np.mean(train_acc)
            model.eval()
            val_acc = evaluate(model, val_loader, batch_size, label_map)
            print("Step: %d, Loss: %.4f, Train Acc: %.2f, Validation Acc: %.2f"%(step, loss.data[0], train_acc, val_acc))
            step_log.append(step)
            loss_log.append(loss.data[0])
            val_acc_log.append(val_acc)
            train_acc_log.append(train_acc)
            model.train()
            train_acc = []
        step += 1
        
    f = open('results_%d.pkl'%num_ep, 'w')
    pickle.dump({'step': step_log, 'loss': loss_log, 'val': val_acc_log, 'train': train_acc_log}, f)
    f.close() 







