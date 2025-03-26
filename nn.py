
import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, GRU, Linear, CTCLoss
import torch.nn.functional as F


class MotifCaller(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):

        super(MotifCaller, self).__init__()

        # CNN Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected Layer
        self.fc = nn.Linear(64, hidden_size)

        # Bidirectional GRU layers
        self.bigru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Sequence Classifier
        self.output = nn.Linear(hidden_size*2, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        
        # CNN layers
        x = self.conv1(x)
        x = torch.relu(x)
        #x = self.norm1(x)
        x = self.pool(x)

        # Apply LayerNorm after permuting the dimensions
        #x = x.permute(0, 2, 1)
        #x = self.norm1(x)

        x = self.conv2(x)
        x = torch.relu(x)

        # Fully connected layer
        x = x.permute(0, 2, 1)
        x = self.fc(x)

        # Bidirectional GRU layers
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        x, _ = self.bigru(x, h0)

        # Output layer
        x = self.output(x)

        #x = x.sum(dim=1)
        return x



class NaiveCaller(nn.Module):
    def __init__(self, input_dim=1, conv_out=128, hidden_dim=128, num_layers=3, num_classes=5):
        super(NaiveCaller, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.fc = nn.Linear(64, hidden_dim)

        self.bigru = GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2)
        
        # Linear layer to output base probabilities
        self.output = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional LSTM

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim) - typically signal data
        """
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Apply LayerNorm
        x = self.conv2(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, conv_out) for LSTM

        x = torch.relu(self.fc(x))
        #x, _ = self.lstm(x)  # LSTM processes sequence
        x, _ = self.bigru(x)
        x = self.output(x)  # Output shape: (batch, seq_len, num_classes)
        #x = x.sum(dim=1)
        return x
        #return F.log_softmax(x, dim=-1)