
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
import torch
from typing import List


class SyntheticDataset(Dataset):

    def __init__(
        self, X: List[any], y: List[any], windows: bool = False,
        window_size: int = 1024, window_step: int = 800):
        
        self.X = X
        self.y = y
        self.windows = windows
        self.window_size = window_size
        self.window_step = window_step

    def split_signal_into_windows(
            self, x: List[float]) -> torch.tensor:
        j = normalize([x]).flatten()
        sequence_length = len(j)
        start_indices = range(
            0, sequence_length - self.window_size + 1, self.window_step)
        windows = [j[start:start + self.window_size] for start in start_indices]
        windows = np.array(windows)
        return torch.tensor(windows, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if self.windows:
            x = self.split_signal_into_windows(
                x)
            
        if isinstance(y, str):
            y = [int(i) for i in y]
            y = torch.tensor(y)

        return x, y