import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,data_path,data_num = 2000) -> None:
        super(MyDataset,self).__init__()
        self.data_path=data_path
        self.data_num= data_num
        self.data = np.load(self.data_path + "/cifar100_features.npy")
        self.label = torch.Tensor(np.load(self.data_path + "/cifar100_labels.npy"))

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx])
        gt = self.label[idx]

        return feature, gt
    
    def __len__(self):
        return self.data.shape[0]