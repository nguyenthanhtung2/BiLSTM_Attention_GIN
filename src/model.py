import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bmm = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, lstm_output, lstm_hidden):
        lstm_hidden = lstm_hidden.squeeze(0)
        lstm_hidden = self.fc(lstm_hidden)
        lstm_hidden = lstm_hidden.unsqueeze(2)
        lstm_output = self.fc(lstm_output)
        attention_scores = self.bmm(lstm_output, lstm_hidden)
        attention_scores = F.softmax(attention_scores, dim=1)
        attention_output = torch.bmm(lstm_output.permute(0, 2, 1), attention_scores).squeeze(2)
        return attention_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GINLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINLayer, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINConv(nn1)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GINLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GINLayer(hidden_dim, hidden_dim))
        self.layers.append(GINLayer(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

class Model(nn.Module):
    def __init__(self, opt) -> None:
        super(Model, self).__init__()

        self.dropout_rate = opt.dropout_rate
        self.upsampling_num = opt.upsampling_num
        self.rnn_size = opt.rnn_size

        self.BiLSTM = BiLSTM(
            input_size=opt.input_size,
            hidden_size=opt.rnn_size,
            num_layers=opt.rnn_layers,
            num_classes=opt.n_classes
        )

        self.attention = Attention(opt.rnn_size)
        self.gin = GIN(
            input_dim=opt.rnn_size,
            hidden_dim=opt.hidden_dim,
            output_dim=opt.rnn_size,
            num_layers=opt.gin_layers
        )
        self.fc = nn.Linear(opt.rnn_size, opt.n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Run through bidirectional LSTM
        rnn_out, _ = self.BiLSTM(input_tensor)

        # Attention module
        H = rnn_out[:, :, :self.rnn_size] + rnn_out[:, :, self.rnn_size:]
        r = self.attention(H)

        # GIN
        new_r = self.gin(r, edge_index)

        h = self.tanh(new_r)

        scores = self.fc(self.dropout(h))

        return scores

