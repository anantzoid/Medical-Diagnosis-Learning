
# coding: utf-8

# In[135]:


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

base_path = '/media/disk3/disk3'
use_cuda = torch.cuda.is_available()


# In[42]:

def read_data_dump(data_path):
    with open(data_path, 'r') as f:
        data = pickle.load(f)
    return data



# In[49]:


from sklearn import preprocessing
import time
def read_embeddings(vecidx_path, vec_path):
    words = []
    t=time.time()
    with open(vecidx_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for i,row in enumerate(tsvreader):
            words.append(row[0])
    
    
    vecs = np.ndarray((len(words), 400))
    with open(vec_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for i,row in enumerate(tsvreader):
            vecs[i,:] = row[:400]
            
    
    vecs = preprocessing.normalize(vecs)
    
    word2vec = {_:vecs[i,:] for i,_ in enumerate(words)}
    
    return word2vec


# In[10]:


def get_labels(ccs_path):
    with open(ccs_path, 'rb') as csvf:
        ccs_mapping = {}
        reader = csv.reader(csvf, delimiter=',')
        _ = next(reader)
        for i, row in enumerate(reader):
            ccs_mapping[row[0]] = row[1]
    return ccs_mapping


print "Starting Reading Stuff..."
_time = time.time()
data = read_data_dump(os.path.join(base_path, 'notes_dump.pkl'))
pretrained = read_embeddings(os.path.join(base_path, 'ri-3gram-400-tsv/vocab.tsv'), 
                                         os.path.join(base_path, 'ri-3gram-400-tsv/vectors.tsv'))

ccs_maping = get_labels(os.path.join(base_path, 'icd9_ccs.csv'))
ccs2idx = {code: i for i, code in enumerate(list(set([ccs_maping[k] for k in ccs_maping.keys()])))}
icd2idx = {icd: ccs2idx[ccs_maping[icd]] for icd in ccs_maping.keys()}

rawdata = []
for key in data:
    if 'notes' in data[key]:
        rawdata.append(data[key])

print "Preprocessing done in %.2f"%((time.time()-_time)/60)
# In[141]:


import random
random.shuffle(rawdata)
margin = int(len(rawdata)*0.8)
testdata = rawdata[:margin]
valdata = rawdata[margin:]


# In[142]:


batch_size = 16
num_workers = 6
hidden_dim = 100

def clean_str_no_stopwords(s):
    s = re.sub('\[\*\*.*\*\*\]|\\n|\s+', ' ', s).replace('  ', ' ').lower().split() 
    return list(set(s))
    #return [token for token in s if token not in stop_words.ENGLISH_STOP_WORDS]

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
        x = sorted(batch[key]['notes'], key=lambda x:datetime.strptime(x['date'], '%Y-%m-%d'))
        x = [clean_str_no_stopwords(note['note']) for note in batch[key]['notes']]
        x = [note for note in x if note != []]
        x = [['pad'] for i in range(max_seq_len-len(x))] + x
        y =  batch[key]['labels']['icd'][0]
        batch_x.append(x)
        batch_y.append(y)
    return (batch_x, batch_y)
train_loader = torch.utils.data.DataLoader(dataset=Dataloader(testdata), batch_size=batch_size, shuffle=True, 
                                                           num_workers=num_workers, collate_fn=padding_collation)
val_loader = torch.utils.data.DataLoader(dataset=Dataloader(valdata), batch_size=batch_size, shuffle=False, 
                                                           num_workers=num_workers, collate_fn=padding_collation)


# In[143]:


class LSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, batch_size):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
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
        x = torch.transpose(x, 1, 0)
        x, _hidden  = self.lstm(x, hidden)
        x = x[-1, :, :].view(self.batch_size, -1)
        x = self.lin(x)
        return x


# In[144]:


hidden_dim = 100
model = LSTMModel(400, hidden_dim, 284, batch_size)
opti = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
crit = nn.CrossEntropyLoss()

if use_cuda:
    model.cuda()
    crit.cuda()


# In[147]:


def evaluate(model, loader, batch_size):
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
        batch_y = torch.from_numpy(np.array([int(icd2idx[_]) for _ in i[1]])).long().view(batch_size)
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
        
        seq_len = len(i[0][0])
        batch_x = np.ndarray((batch_size, seq_len, 400))
        for bs, adm in enumerate(i[0]):
            for idx, note in enumerate(adm):
                note_vec = []
                for token in note:
                    note_vec.append(pretrained.get(re.sub(r'[^\w\s]','', token), pretrained['unknown']))
                
                batch_x[bs, idx, :] = np.mean(note_vec)  
                # batch_x[bs, idx, :] = np.mean([pretrained.get(re.sub(r'[^\w\s]','', token), pretrained['unknown']) for token in note], axis=0)
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(np.array([int(icd2idx[_]) for _ in i[1]])).long().view(batch_size)
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
            val_acc = evaluate(model, val_loader, batch_size)
            print("Step: %d, Loss: %.4f, Train Acc: %.2f, Validation Acc: %.2f"%(step, loss.data[0], train_acc, val_acc))
            step_log.append(step)
            loss_log.append(loss.data[0])
            val_acc_log.append(val_acc)
            train_acc_log.append(train_acc)
            model.train()
            train_acc = []
        step += 1
        
    f=open('results_%d.pkl'%num_ep, 'w')
    pickle.dump({'step': step_log, 'loss': loss_log, 'val': val_acc_log, 'train': train_acc_log}, f)
    f.close() 







