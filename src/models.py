import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

class TextData(data.Dataset):
    def __init__(self, data):
        super(TextData, self).__init__()
        self.data = data
    def __getitem__(self, index):

        #return (self.data[index]['text'], self.data[index]['label'])
        return (self.data[index]['text_index_sequence'], self.data[index]['label'])
    def __len__(self):
        return len(self.data)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, labels, batch_size, use_cuda):
        super(LSTMModel, self).__init__()
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.lin = nn.Linear(2 * hidden_dim, len(labels.keys()))
        self.init_hidden()

    def init_hidden(self):
        hidden1 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
        hidden2 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
        if self.use_cuda:
            self.hidden = (hidden1.cuda(), hidden2.cuda())
        else:
            self.hidden = (hidden1, hidden2)

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, 1, 0)
        x, _hidden = self.lstm(x, self.hidden)
        x = x[-1, :, :].view(self.batch_size, -1)
        x = self.lin(x)
        return x

class FastText(nn.Module):
    """
    FastText model
    """
    def __init__(self, vocab_size, emb_dim, out_dim, cuda):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(FastText, self).__init__()
        self.cuda = cuda
        self.embed = nn.Embedding(vocab_size + 2, emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, out_dim)
        self.init_weights()

    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        length = length.type(torch.FloatTensor).unsqueeze(1)
        if self.cuda:
          length = length.cuda()
        out = torch.div(out, length)
        out = self.linear1(out)
        return out

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.linear1.weight.data.uniform_(-init_range, init_range)
        self.linear1.bias.data.fill_(0)
