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
        # print(x.data.shape)
        x = torch.transpose(x, 1, 0)
        # print(x.data.shape)
        x, _hidden = self.lstm(x, self.hidden)
        # print(x.data.shape)
        x = x[-1, :, :].view(self.batch_size, -1)
        # print(x.data.shape)
        x = self.lin(x)
        # print(x.data.shape)
        return x
