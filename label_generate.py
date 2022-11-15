import faiss
import torch
import torch.utils.data as data
import numpy as np
import time

# 重写
class ReassignedDataset(data.Dataset):
    def __init__(self, data_indexes, pseudolabels, dataset, transform=None):
        self.data = dataset
        self.data_indexes = data_indexes
        self.pseudolabels = torch.Tensor(pseudolabels).long()

    def __getitem__(self, index):
        data = self.data[self.data_indexes[index]]
        plabel = self.pseudolabels[index]
        return data, plabel

    def __len__(self):
        return len(self.data_indexes)
# 重写
def cluster_assign(cluster_lists, dataset):
    assert cluster_lists is not None
    pseudolabels = []
    data_indexes = []
    for cluster, data in enumerate(cluster_lists):
        data_indexes.extend(data)
        pseudolabels.extend([cluster] * len(data))

    return ReassignedDataset(data_indexes, pseudolabels, dataset)

def run_kmeans(x, nmb_clusters, verbose=False):

    n_data, d = x.shape
    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config) # 初始化index，GPU资源，d为维度

    # perform the training
    clus.train(x, index) 
    _, I = index.search(x, 1) # search API，搜索距离x每个数据最近的1个index中的向量
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        end = time.time()

        # cluster the data
        I, loss = run_kmeans(data, self.k, verbose)
        self.cluster_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.cluster_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
