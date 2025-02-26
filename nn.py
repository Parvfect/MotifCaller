
import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, GRU, Linear, CTCLoss
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        self.layer_1 = Conv1d(
            in_channels=1, out_channels=16, stride=2, kernel_size=5)
        self.layer_2 = Conv1d(
            in_channels=4, out_channels=16, stride=2, kernel_size=5)
        self.layer_3 = Conv1d(
            in_channels=16, out_channels=64, stride=2, kernel_size=5)
        self.layer_4 = Conv1d(
            in_channels=64, out_channels=256, stride=2, kernel_size=19)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        #x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = x.permute(0, 2, 1)
        return x
    

class RNNLayers(nn.Module):
    def __init__(self, hidden_size, num_layers, n_classes):
        super(RNNLayers, self).__init__()

        self.bigru = GRU(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True)

        self.output = Linear(hidden_size*2, n_classes)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    
    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        x, _ = self.bigru(x, h0)

        return F.log_softmax(self.output(x), dim=2)


class MotifCaller(nn.Module):
    """Batch size 1 fixed"""
    def __init__(self, hidden_size=256, n_layers=3, n_classes=20):
        super(MotifCaller, self).__init__()
        self.cnn_layers = ConvolutionalEncoder()
        self.rnn_layers = RNNLayers(
            hidden_size=256, num_layers=n_layers, n_classes=n_classes)

    def forward(self, x):
        x = self.cnn_layers.forward(x)
        x = self.rnn_layers.forward(x)
        return x
