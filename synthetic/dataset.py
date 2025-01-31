import os
import torch
import numpy as np
from torch.utils.data import Dataset
from gen_dataset import gen_da_data_ortho
import ipdb as pdb

class DANS(Dataset):
    def __init__(self, dataset):
        super().__init__()
        # self.path = os.path.join(directory, dataset, "data.npz")
        # self.npz = np.load(self.path)
        self.data = dataset

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        y = torch.from_numpy(self.data["y"][idx].astype('float32'))
        x = torch.from_numpy(self.data["x"][idx].astype('float32'))
        c = torch.from_numpy(self.data["c"][idx, None].astype('float32'))
        sample = {"y": y, "x": x, "c": c}
        return sample


class DANS_joint(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        z = torch.from_numpy(self.data["z"][idx].astype('float32'))
        x = torch.from_numpy(self.data["x"][idx].astype('float32'))
        u = torch.from_numpy(self.data["u"][idx,None].astype('float32')).squeeze().to(torch.int64)
        y = torch.from_numpy(self.data["y"][idx,None].astype('float32')).squeeze().to(torch.int64)
        sample = {"z": z, "x": x, "u": u, "y": y}
        return sample
