# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, size=10, length=1):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], 1

    def __len__(self):
        return self.len
