import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,data_path,data_num = 2000) -> None:
        super(MyDataset,self).__init__()
        self.data_path=data_path
        self.data_num= data_num
        data = np.load(self.data_path + "/cifar100_features.npy")
        sampled_idx = np.random.choice(data.shape[0], self.data_num, replace=False)
        self.data = data[sampled_idx]
        labels = np.load(self.data_path + "/cifar100_labels.npy")
        self.labels = torch.Tensor(labels[sampled_idx])

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.data[idx])
        gt = self.labels[idx]

        return feature, gt
    
    def __len__(self):
        return self.data.shape[0]

class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)