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
use_cuda = torch.cuda.is_available()




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
        x = x.contiguous().view(true_x_size[2], self.batch_size*true_x_size[1], self.embed_dim)
        return self.word_rnn(x, _hidden)
        
   
    def init_hidden(self, b_size, volatile=False):
        hidden1 = Variable(torch.zeros(2, b_size,  self.hidden_dim), volatile=volatile)
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
        attends = torch.mul(attends, self.context)
        #print("attention_combine:", attn_combine.size())        
        attends = self.sm(attends.contiguous().view(-1, self.hidden_dim))
        #print("sm size:", alpha.size())
        #print(x.size())
        attended = torch.mul(x, attends).contiguous().view(self.batch_size, sentence_size, -1, self.hidden_dim)
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
    def init_hidden(self, volatile=False):
        hidden1 = Variable(torch.zeros(2, self.batch_size,  self.hidden_dim), volatile=volatile)
        #hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return hidden1#, hidden2)      

class Classifer(nn.Module):
    def __init__(self, hidden_dim, op_dim):
        super(Classifer, self).__init__()
        self.lin = nn.Linear(hidden_dim, op_dim)
    def forward(self, x):
        return self.lin(x)    


# In[12]:

class Ensemble(nn.Module):
    def __init__(self, embed_dim, vocabulary_size, hidden_dim, batch_size, label_map):
        super(Ensemble, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_rnn = WordModel(embed_dim, vocabulary_size, hidden_dim, batch_size)
        self.wordattention = Attend(batch_size, 2*hidden_dim)
        self.sent_rnn = SentModel(batch_size, 2*hidden_dim)
        self.sentattention = Attend(batch_size, 4*hidden_dim)
        self.clf = Classifer(4*hidden_dim, len(label_map.keys()))

    def forward(self, x, word_hidden, sent_hidden):
        #print("raw size:", batch_x.size())
        true_batch_size = x.size()
        x, hidden = self.word_rnn(x, word_hidden)
        #print("word rnn op size:", x.size())
        #print("word rnn hidden size:", hidden.size())    
        x = x.contiguous().view(true_batch_size[2], true_batch_size[0]*true_batch_size[1], -1) # sent_size x batch_size x 2*hidd
        #print("============")
        #print("word attention ip size:", x.size())
        x = self.wordattention(x, true_batch_size[1]) # batch_size x sent_size x 2*hidden
        #print(sent_op())
        #print("============")    
        x, sent_hidden = self.sent_rnn(x, sent_hidden)
        #print("sent rnn op size:", sent_op.size())
        x = x.contiguous().view(true_batch_size[1], self.batch_size, -1) # sent_size x batch_size x 2*hidden
        x = self.sentattention(x, 1)
        x = x.contiguous().view(self.batch_size, 4*self.hidden_dim)
        x = self.clf(x)
        return x


def xavier_weight_init(m):
    classname = m.__class__.__name__
    torch.nn.init.xavier_uniform(m.weight)
