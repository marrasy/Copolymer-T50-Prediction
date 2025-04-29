import torch
import torch.nn as nn
import torch.nn.functional as F
#我之后加上的
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.linear(x[:, -1, :])
        return x
    
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        self.embed_size = hidden_size
        self.values = nn.Linear(input_size, hidden_size, bias=False)
        self.keys = nn.Linear(input_size, hidden_size, bias=False)
        self.queries = nn.Linear(input_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        attention = torch.bmm(queries, keys.transpose(1, 2)) / self.embed_size ** 0.5
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(attention, values)
        out = self.fc_out(out)
        return out
    
class ATTN(nn.Module): # Global
    def __init__(self, input_size, hidden_size, output_size):
        super(ATTN, self).__init__()
        self.attn1 = SelfAttention(input_size, hidden_size, hidden_size)
        self.attn2 = SelfAttention(hidden_size, hidden_size, hidden_size)
        self.attn3 = SelfAttention(hidden_size, hidden_size, hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.attn3(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.pool(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
