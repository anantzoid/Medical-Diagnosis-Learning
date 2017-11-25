
# coding: utf-8

# In[6]:


import csv
import re
import random
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import math
import spacy
nlp = spacy.load('en')
from spacy.en import English
tokenizer = English().Defaults.create_tokenizer(nlp)
use_cuda = torch.cuda.is_available()


# In[ ]:

'''
notedata = []
with open('../data/mimic/NOTEEVENTS.csv', 'r') as f:
    rea = csv.reader(f, delimiter=',', quotechar='"')
    _ = next(rea)
    for _,row in enumerate(rea):
        if row[6].lower() == "discharge summary":
            notedata.append([re.sub("\d", "d", row[-1]), row[2]])

'''

# In[7]:


def get_top_labels(path):
    labels = [row[0] for row in  csv.reader(open(path, "r"), delimiter=",")]
    return {i:_ for _,i in enumerate(labels)}

def load_summaries(path):
    data = []
    for row in  csv.reader(open(path, "r"), delimiter=",", quotechar='"'):
        if row[1] == '':
            continue
        data.append({
            'text': re.sub("\d", "d", row[1]),
            'label': row[2].split(',')
        })
    return data


# In[8]:



def build_dictionary(data, PADDING, UNKNOWN, vocab_threshold, tokenizer):    
    all_tokens = []
    for doc in tokenizer.pipe([_['text'] for _ in data], batch_size=50):
        all_tokens.extend([str(_) for _ in list(doc)])
    word_counter = Counter(all_tokens)
    vocabulary = list(set([word for word in word_counter if word_counter[word] > vocab_threshold]))
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    return (word_indices, vocabulary)

class TextData(data.Dataset):
    def __init__(self, data, token2idx, UNKNOWN, label_map):
        super(TextData, self).__init__()
        for i, row in enumerate(data):
            tokens = nlp(row['text'])            
            sentences = [str(s).replace("\n", "") for s in tokens.sents]                                        
            data[i]['text_index_sequence'] = [[token2idx.get(str(word), token2idx[UNKNOWN]) for word in
                                                           list(tokenizer(sent))] for sent in sentences]                                    
            label_onehot = np.zeros((len(label_map.keys())))
            for la in row['label']:
                label_onehot[label_map[la]] = 1
            data[i]['label'] = label_onehot        

        self.data = data
        
    def __getitem__(self, index):
        return (self.data[index]['text_index_sequence'], self.data[index]['label'])
    def __len__(self):
        return len(self.data)

def sent_batch_collate(batch):
    max_note_len = max([len(_[0]) for _ in batch])
    max_sentence_len = max([len(i) for _ in batch for i in _[0]])
    
    x = torch.zeros(len(batch), max_note_len, max_sentence_len)
    for n,note in enumerate(batch):
        for s,sentence in enumerate(note[0]):
            for w, word in enumerate(sentence):                                
                x[n, s, w] = float(word)

    return (x.long(), torch.from_numpy(np.array([_[1] for _ in batch])).float())


# In[9]:


label_path = '../data/top50_labels.csv'
label_map = {i:_ for _,i in enumerate(get_top_labels(label_path))}

data_path = '../data/summaries_labels.csv'
training_set = load_summaries(data_path)

random.shuffle(training_set)
testset = training_set[int(len(training_set)*0.9):]
training_set = training_set[:int(len(training_set)*0.9)]
print("Size of train: %d"%len(training_set))

# In[10]:


PADDING = "<PAD>"
UNKNOWN = "<UNK>"
max_seq_length = 500
min_vocab_threshold = 5
batch_size = 32
num_workers = 4
embed_dim = 50
hidden_dim = 100
lr = 1e-3
num_epochs = 10


token2idx, vocabulary = build_dictionary(training_set, PADDING, UNKNOWN, min_vocab_threshold, tokenizer)
print("Vocab size: %d"%len(vocabulary))
dataset= TextData(training_set, token2idx, UNKNOWN, label_map)
train_loader = torch.utils.data.DataLoader(dataset= dataset, batch_size=batch_size, shuffle=True,
                                                           num_workers=num_workers, collate_fn=sent_batch_collate)
val_loader = torch.utils.data.DataLoader(dataset= TextData(testset, token2idx, UNKNOWN, label_map), batch_size=batch_size, shuffle=True,
                                                           num_workers=num_workers, collate_fn=sent_batch_collate)


# In[11]:


class WordModel(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim, batch_size):
        super(WordModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.word_rnn = nn.GRU(embed_dim, hidden_dim,  bidirectional=True)
                
    def forward(self, x, _hidden):
        true_x_size = x.size()
        x = x.view(self.batch_size, -1)
        #print("before embedding", x.size())        
        x = self.word_embed(x)        
        #print("after embedding", x.size())
        x = torch.transpose(x, 1, 0)
        return self.word_rnn(x, _hidden)
        
#         descriptors = []#torch.zeros(x.size(-2), _hidden[0].size(0)*2, _hidden[0].size(1), _hidden[0].size(2))
#         for w in range(x.size(0)):
#             #print("seq size:", x[w, :, :].size())
#             op, _hidden  = self.word_rnn(x[w, :, :].contiguous().view(1, self.batch_size, -1), _hidden)
#             descriptors.append(torch.cat([_hidden[0], _hidden[1]], 2))
            
#         descriptors = torch.cat(descriptors, 0)
#         #descriptors = torch.transpose(descriptors, 1, 0)#.contiguous().view(batch_size, -1, x.size(0),2*hidden_dim)
        
#         #x = x[-1, :, :].view(self.batch_size, -1)        
#         #print("desc size:", descriptors.size())
#         print("op full size:", op.size())
#         return (op, descriptors)#[-1, :, :]
    
    def init_hidden(self):
        hidden1 = Variable(torch.zeros(2, self.batch_size,  self.hidden_dim))
        #hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return hidden1#, hidden2)

    
class Attend(nn.Module):
    def __init__(self, batch_size, hidden_dim):
        super(Attend, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        
        self.context = Variable(torch.FloatTensor(hidden_dim))        
        stdv = 1. / math.sqrt(self.context.size(0))
        self.context.data.uniform_(-stdv, stdv)
        
        self.sm = nn.Softmax()
    def forward(self, x, sentence_size):
        attends = []
        for i in range(x.size(0)):
            #print(x[i,:,:].size())
            attends.append(F.tanh(self.lin(x[i,:,:])).unsqueeze(0))
        #print ("single attend:", attends[0].size())
        attends = torch.cat(attends)
        #print("cat attention:", attends.size())
        attn_combine = torch.mul(attends, self.context)
        #print("attention_combine:", attn_combine.size())        
        alpha = self.sm(attn_combine.contiguous().view(-1, self.hidden_dim))
        #print("sm size:", alpha.size())
        #print(x.size())
        attended = torch.mul(x, alpha).contiguous().view(self.batch_size, sentence_size, -1, self.hidden_dim)
        #print("x.alpha prod:", attended.size())
        attended = torch.sum(attended, 2)
        #print("attended sum:", attended.size())
        return attended

class SentModel(nn.Module):
    def __init__(self, batch_size, hidden_dim):
        super(SentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sent_rnn = nn.GRU(hidden_dim, hidden_dim,  bidirectional=True)
    def forward(self, x, _hidden):
        x = torch.transpose(x, 1, 0)
        return self.sent_rnn(x, _hidden)
    def init_hidden(self):
        hidden1 = Variable(torch.zeros(2, self.batch_size,  self.hidden_dim))
        #hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return hidden1#, hidden2)      

class Classifer(nn.Module):
    def __init__(self, hidden_dim, op_dim):
        super(Classifer, self).__init__()
        self.lin = nn.Linear(hidden_dim, op_dim)
    def forward(self, x):
        return self.lin(x)    


# In[12]:


model = WordModel(embed_dim, len(vocabulary), hidden_dim, batch_size)
wordattention = Attend(batch_size, 2*hidden_dim)
sent_rnn = SentModel(batch_size, 2*hidden_dim)
sentattention = Attend(batch_size, 4*hidden_dim)
clf = Classifer(4*hidden_dim, len(label_map.keys()))
crit = nn.BCEWithLogitsLoss()
all_params = list(model.parameters()) + list(wordattention.parameters()) +     list(sent_rnn.parameters()) + list(sentattention.parameters())+ list(clf.parameters())
opti = torch.optim.Adam(all_params, lr=lr, betas=(0.5, 0.999))

if use_cuda:
    model.cuda()
    wordattention.cuda()
    sent_rnn.cuda()
    sentattention.cuda()
    clf.cuda()
    crit.cuda()
    wordattention.context = wordattention.context.cuda()
    sentattention.context = sentattention.context.cuda()

# In[13]:
print("Starting training...")
step = 0
for n_e in range(num_epochs):
    for batch in train_loader:
        model.zero_grad()
        wordattention.zero_grad()
        sent_rnn.zero_grad()
        sentattention.zero_grad()
        clf.zero_grad()

        batch_x = Variable(batch[0])
        batch_y = Variable(batch[1])        
            
        _hidden = model.init_hidden()
        sent_hidden = sent_rnn.init_hidden()
        if use_cuda:
            batch_x, batch_y, _hidden, sent_hidden = batch_x.cuda(), batch_y.cuda(), _hidden.cuda(), sent_hidden.cuda()
        #print("raw size:", batch_x.size())
        x, hidden = model(batch_x, _hidden)
        #print("word rnn op size:", x.size())
        #print("word rnn hidden size:", hidden.size())    
        x = x.contiguous().view(batch_x.size(2), batch_x.size(0)*batch_x.size(1), -1) # sent_size x batch_size x 2*hidd
        #print("============")
        #print("word attention ip size:", x.size())
        sentence_reprs = wordattention(x, batch_x.size(1)) # batch_size x sent_size x 2*hidden
        #print(sentence_reprs.size())
        #print("============")    
        sent_op, sent_hidden = sent_rnn(sentence_reprs, sent_hidden)
        #print("sent rnn op size:", sent_op.size())
        sent_op = sent_op.contiguous().view(batch_x.size(1), batch_size, -1) # sent_size x batch_size x 2*hidden
        sent_att = sentattention(sent_op, 1)
        sent_att = sent_att.contiguous().view(batch_size, 4*hidden_dim)
        pred_prob = clf(sent_att)
        loss = crit(pred_prob, batch_y)
        loss.backward()
        opti.step()
        if step%100 ==0:
            val_loss = []
            for val_batch in val_loader:
                batch_x = Variable(batch[0])
                batch_y = Variable(batch[1])
                _hidden = model.init_hidden()
                sent_hidden = sent_rnn.init_hidden()
                if use_cuda:
                    batch_x, batch_y, _hidden, sent_hidden = batch_x.cuda(), batch_y.cuda(), _hidden.cuda(), sent_hidden.cuda()
                
                x, hidden = model(batch_x, _hidden)
                x = x.contiguous().view(batch_x.size(2), batch_x.size(0)*batch_x.size(1), -1)
                sentence_reprs = wordattention(x, batch_x.size(1))
                sent_op, sent_hidden = sent_rnn(sentence_reprs, sent_hidden)
                sent_op = sent_op.contiguous().view(batch_x.size(1), batch_size, -1)
                sent_att = sentattention(sent_op, 1)
                sent_att = sent_att.contiguous().view(batch_size, 4*hidden_dim)
                pred_prob = clf(sent_att)
                val_loss = crit(pred_prob, batch_y)
                val_loss.append(val_loss.data[0])
            print("Epoch: %d, Step: %d, Train Loss: %.2f, Val Loss: %.2f"%(n_e, step, loss.data[0], np.mean(val_loss)))
        step += 1
        


