import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class GPS_CNN(nn.Module):
    def __init__(self, num_bins=51):
        super().__init__()
        self.num_bins = num_bins
        self.conv1 = nn.Conv1d(
            in_channels=42, out_channels=8, kernel_size=5, padding="same"
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=4, kernel_size=5, padding="same"
        )
        self.fc1 = nn.Linear(in_features=212, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        # apply conv1d and relu activation
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        # flatten
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # apply linear layers and relu activations
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GPS_CNN_Dataset(torch.utils.data.Dataset):
    def __init__(self, X: str, y: str, device=None):
        # read in numpy file for data
        X = np.load(X)
        # read in numpy file for labels
        y = np.load(y)
        # Currently Data is in Shape (num_samples, num_bins, 43) reshape to (num_samples, 43, num_bins)
        X = np.swapaxes(X, 1, 2)
        # Load the Data into Tensors
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)

        print("X Shape: ", self.X.shape)
        print("y Shape: ", self.y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y
