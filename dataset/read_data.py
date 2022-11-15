import numpy as np
import os
import pickle

# root_path = "E:/dataset/feature"
# sub_paths = ['CIFAR10-MCR2', 'CIFAR100-MCR2', 'EMNIST', 'FashionMNIST', 'MNIST', 'REUTERS10K-IDF', 'STL10-MCR2']

# for sub_path in sub_paths:
#     path = os.path.join(root_path,sub_path)
#     files = os.listdir(path)
#     for file in files:
#         if file.split(".")[-1] == 'npy':
#             data = np.load(os.path.join(path,file),allow_pickle=True)
#             print(file+':'+str(data.shape))
#         elif file.split(".")[-1] == 'pkl':
#             with open(os.path.join(path,file),'rb') as f:
#                 data = pickle.load(f)
#             print(file+':'+str(data.shape))

data = np.load('E:/dataset/数据集特征/CIFAR100-MCR2/labels.npy')
print(type(data[0]))
