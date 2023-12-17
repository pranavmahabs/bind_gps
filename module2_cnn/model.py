import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np


class GPS_CNN(nn.Module):
    def __init__(self, num_bins=21, num_channels=24):
        super().__init__()
        self.num_bins = num_bins
        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=8, kernel_size=5, padding="same"
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=4, kernel_size=5, padding="same"
        )
        self.fc1 = nn.Linear(in_features=92, out_features=64)
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


class GPS_MLP(nn.Module):
    def __init__(self, num_channels=24):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_channels, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=2)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # apply linear layers and relu activations
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class GPS_MLP_Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device=None):
        self.positive_indices = [i for i, x in enumerate(X) if y[i] == 1]
        self.negative_indices = [i for i, x in enumerate(X) if y[i] == 0]
        self.num_positive = len(self.positive_indices)
        self.num_negative = len(self.negative_indices)
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)

        print("X Shape: ", self.X.shape)
        print("y Shape: ", self.y.shape)

    def __len__(self):
        return len(self.X)
    
    def get_quantities(self):
        print("Num Positive: {}, Num Negative {}".format(self.num_positive, self.num_negative))

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y

class GPS_CNN_Dataset(Dataset):
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
