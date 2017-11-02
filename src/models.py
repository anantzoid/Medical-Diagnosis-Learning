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
        return (self.data[index]['text_index_sequence'].view(-1), self.data[index]['label'])
    def __len__(self):
        return len(self.data)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, labels, batch_size):
        super(LSTMModel, self).__init__()
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, len(labels.keys()))
    def init_hidden(self):
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)), Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
    def forward(self, x, hidden):
        x = self.embed(x)
        x = torch.transpose(x, 1, 0)
        x, _hidden  = self.lstm(x, hidden)
        x = x[-1, :, :].view(self.batch_size, -1)
        x = self.lin(x)
        return x
