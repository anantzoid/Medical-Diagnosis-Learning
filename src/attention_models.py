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


class CBOW(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim, batch_size):
        super(CBOW, self).__init__()
        self.batch_size = batch_size
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.init_weights()

    def forward(self, data, length):
        print(data.size())
        out = self.word_embed(data)
        print(out.size())
        out = torch.sum(out, dim=1)
        out = torch.div(out, length)
        return out

    def init_weights(self):
        init_range = 0.1
        self.word_embed.weight.data.uniform_(-init_range, init_range)

    
    def init_hidden(self, b_size, volatile=False):
        # Not needed for this model, does the same as the word model 
        # so training code does not have to be changed
        hidden1 = Variable(torch.zeros(
            2, b_size,  self.hidden_dim), volatile=volatile)
        return hidden1

class WordModel(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_dim, batch_size):
        super(WordModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.word_rnn = nn.GRU(embed_dim, hidden_dim,  bidirectional=True)
        self.init_embedding()

    def forward(self, x, _hidden):
        true_x_size = x.size()
        # print("True x size: {}".format(x.data.size()))
        # print("Hidden size: {}")
        x = x.view(self.batch_size, -1)
        # print("before embedding", x.data.size())
        x = self.word_embed(x)
        # print("after embedding", x.data.size())
        # Stack to (batch_size x sentence size) x num words per sentence x embed_dim
        x = x.contiguous().view(
            true_x_size[0] * true_x_size[1], true_x_size[2], self.embed_dim)
        x = torch.transpose(x, 1, 0)
        # print("After reshape for GRU: {}".format(x.data.size()))
        return self.word_rnn(x, _hidden)

    def init_hidden(self, b_size, volatile=False):
        hidden1 = Variable(torch.zeros(
            2, b_size,  self.hidden_dim), volatile=volatile)
        #hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return hidden1  # , hidden2)

    def init_embedding(self):
        init_range = 0.1
        self.word_embed.weight.data.uniform_(-init_range, init_range)


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

    def forward(self, x):
        #print("x: ", x.size())
        seq_length = x.size(0)
        #print("sequence length: ", seq_length)
        x = torch.transpose(x, 1, 0)
        #print("x transposed: ", x.size())
        attends = []
        for i in range(x.size(0)):
            #print(x[i,:,:].size())
            attends.append(F.tanh(self.lin(x[i, :, :])).unsqueeze(0))
        #print ("single attend:", attends[0].size())
        # Reshape for dot product of each element of each sentence with context vector
        attends = torch.cat(attends).view(-1, self.hidden_dim)
        #print("cat attention:", attends.size())
        # Reshaped back to batch size x sentence length
        similarity = torch.matmul(attends, self.context).view(-1, seq_length)
        #print("similarity:", similarity.size())
        probs = self.sm(similarity).unsqueeze(2)
        #print("probs:", probs.size())
        attended = torch.mul(x, probs)
        #print("attended:", attended.size())
        attended = torch.sum(attended, 1)
        #print("final attended:", attended.size())

        # #print("attention_combine:", attn_combine.size())
        # # attends = torch.mul(attends, self.context)
        # #print("attention_combine:", attn_combine.size())
        # attends = self.sm(attends.contiguous().view(-1, self.hidden_dim))
        # #print("sm size:", alpha.size())
        # # print(x.size())
        # attended = torch.mul(x, attends).contiguous().view(
        #     self.batch_size, sentence_size, -1, self.hidden_dim)
        # #print("x.alpha prod:", attended.size())
        # attended = torch.sum(attended, 2)
        # #print("attended sum:", attended.size())
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
        hidden1 = Variable(torch.zeros(2, self.batch_size,
                                       self.hidden_dim), volatile=volatile)
        #hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return hidden1  # , hidden2)


class Classifer(nn.Module):
    def __init__(self, hidden_dim, op_dim):
        super(Classifer, self).__init__()
        self.lin = nn.Linear(hidden_dim, op_dim)

    def forward(self, x):
        return self.lin(x)


class CBOWSentModel(nn.Module):
    def __init__(self, embed_dim, vocabulary_size, hidden_dim, batch_size, label_map):
        super(CBOWSentModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_rnn = CBOW(
            embed_dim, vocabulary_size, hidden_dim, batch_size)
        self.sent_rnn = SentModel(batch_size, 2 * hidden_dim)
        self.clf = Classifer(4 * hidden_dim, len(label_map.keys()))
        # self.clf.apply(xavier_weight_init)

    def forward(self, x, word_hidden, sent_hidden, length_x):
        # print("raw size:", x.size())
        # print("Word hidden: ", word_hidden.size())
        # print("Sent hidden: ", sent_hidden.size())
        # batch_size x sentence size x num words per sentence
        true_batch_size = x.size()
        # print("======= word model =====")
        x = self.word_rnn(x, length_x)
        # print("word rnn op size:", x.size())
        # print("Sentence summary shape:", x.size())
        # Output: (batch size * num sentences) x hidden state size
        # Reshape to: batch size x num sentences x hidden state size
        x = x.contiguous().view(
            true_batch_size[0], true_batch_size[1], -1)
        # print("bs x sent x hidden:", x.size())
        # print("======= sentence model =====")
        x, sent_hidden = self.sent_rnn(x, sent_hidden)
        x = x[-1].squeeze()
        x = x.contiguous().view(true_batch_size[0], -1)
        # print("After sentence model", x.size())
        x = self.clf(x)
        # print("Output size: ", x.size())
        return x


class WordSentModel(nn.Module):
    def __init__(self, embed_dim, vocabulary_size, hidden_dim, batch_size, label_map):
        super(WordSentModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_rnn = WordModel(
            embed_dim, vocabulary_size, hidden_dim, batch_size)
        self.sent_rnn = SentModel(batch_size, 2 * hidden_dim)
        self.clf = Classifer(4 * hidden_dim, len(label_map.keys()))
        # self.clf.apply(xavier_weight_init)

    def forward(self, x, word_hidden, sent_hidden, length_x):
        # print("raw size:", x.size())
        # print("Word hidden: ", word_hidden.size())
        # print("Sent hidden: ", sent_hidden.size())
        # batch_size x sentence size x num words per sentence
        true_batch_size = x.size()
        # print("======= word model =====")
        x, hidden = self.word_rnn(x, word_hidden)
        # print("word rnn op size:", x.size())
        # print("word rnn hidden size:", word_hidden.size())
        # Select last element from sequence
        x = x[-1].squeeze()
        # print("Sentence summary shape:", x.size())
        # Output: (batch size * num sentences) x hidden state size
        # Reshape to: batch size x num sentences x hidden state size
        x = x.contiguous().view(
            true_batch_size[0], true_batch_size[1], -1)
        # print("bs x sent x hidden:", x.size())
        # print("======= sentence model =====")
        x, sent_hidden = self.sent_rnn(x, sent_hidden)
        x = x[-1].squeeze()
        x = x.contiguous().view(true_batch_size[0], -1)
        # print("After sentence model", x.size())
        x = self.clf(x)
        # print("Output size: ", x.size())
        return x


class HANModel(nn.Module):
    def __init__(self, embed_dim, vocabulary_size, hidden_dim, batch_size, label_map):
        super(HANModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_rnn = WordModel(
            embed_dim, vocabulary_size, hidden_dim, batch_size)
        self.wordattention = Attend(batch_size, 2 * hidden_dim)
        self.sent_rnn = SentModel(batch_size, 2 * hidden_dim)
        self.sentattention = Attend(batch_size, 4 * hidden_dim)
        self.clf = Classifer(4 * hidden_dim, len(label_map.keys()))
        # self.clf.apply(xavier_weight_init)

    def forward(self, x, word_hidden, sent_hidden, length_x):
        #print("raw size:", x.size())
        #print("Word hidden: ", word_hidden.size())
        #print("Sent hidden: ", sent_hidden.size())
        # batch_size x sentence size x num words per sentence
        true_batch_size = x.size()
        #print("======= word model =====")
        x, hidden = self.word_rnn(x, word_hidden)
        #print("word rnn op size:", x.size())
        #print("word rnn hidden size:", word_hidden.size())
        # Attend on words
        x = self.wordattention(x)
        #print("Sentence summary shape:", x.size())
        # Output: (batch size * num sentences) x hidden state size
        # Reshape to: batch size x num sentences x hidden state size
        x = x.contiguous().view(
            true_batch_size[0], true_batch_size[1], -1)
        #print("bs x sent x hidden:", x.size())
        #print("======= sentence model =====")
        x, sent_hidden = self.sent_rnn(x, sent_hidden)
        # Attend on sentences
        x = self.sentattention(x)
        x = x.contiguous().view(true_batch_size[0], -1)
        #print("After sentence model", x.size())
        x = self.clf(x)
        #print("Output size: ", x.size())
        #exit()
        return x



class Ensemble(nn.Module):
    def __init__(self, embed_dim, vocabulary_size, hidden_dim, batch_size, label_map):
        super(Ensemble, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_rnn = WordModel(
            embed_dim, vocabulary_size, hidden_dim, batch_size)
        self.wordattention = Attend(batch_size, 2 * hidden_dim)
        self.sent_rnn = SentModel(batch_size, 2 * hidden_dim)
        self.sentattention = Attend(batch_size, 4 * hidden_dim)
        self.clf = Classifer(4 * hidden_dim, len(label_map.keys()))

    def forward(self, x, word_hidden, sent_hidden, length_x):
        #print("raw size:", batch_x.size())
        true_batch_size = x.size()
        x, hidden = self.word_rnn(x, word_hidden)
        #print("word rnn op size:", x.size())
        #print("word rnn hidden size:", hidden.size())
        # sent_size x batch_size x 2*hidd
        x = x.contiguous().view(
            true_batch_size[2], true_batch_size[0] * true_batch_size[1], -1)
        # print("============")
        #print("word attention ip size:", x.size())
        # batch_size x sent_size x 2*hidden
        x = self.wordattention(x, true_batch_size[1])
        # print(sent_op())
        # print("============")
        x, sent_hidden = self.sent_rnn(x, sent_hidden)
        #print("sent rnn op size:", sent_op.size())
        # sent_size x batch_size x 2*hidden
        x = x.contiguous().view(true_batch_size[1], self.batch_size, -1)
        x = self.sentattention(x, 1)
        x = x.contiguous().view(self.batch_size, 4 * self.hidden_dim)
        x = self.clf(x)
        return x


def xavier_weight_init(m):
    classname = m.__class__.__name__
    if classname != 'GRU':
        torch.nn.init.xavier_uniform(m.weight)
