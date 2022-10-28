import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
	def __init__(self, input_dims, hid_dims, out_dims):
		super(MLP, self).__init__()
		self.input_dims = input_dims
		self.hid_dims = hid_dims
		self.output_dims = out_dims
		self.layer1 = nn.Linear(self.input_dims,self.hid_dims[0])
		self.layer2 = nn.Linear(self.hid_dims[0],self.hid_dims[1])
		self.layer3 = nn.Linear(self.hid_dims[1],self.output_dims)


	def forward(self,x):
		out1 = nn.ReLU(self.layer1(x))
		out2 = nn.ReLU(self.layer2(out1))
		out3 = nn.ReLU(self.layer2(out2))

		out = out3 + out1

		return out

class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))
    
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)

class SENet(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims)

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb
    
    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out