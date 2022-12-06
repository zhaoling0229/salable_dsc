import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layer1 = nn.Linear(self.input_dims, self.hid_dims[0])
        self.layer2 = nn.Linear(self.hid_dims[0], self.hid_dims[1])
        self.layer3 = nn.Linear(self.hid_dims[1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()
    
    def reset_parameters(self):
        for layer in [self.layer1,self.layer2]:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.layer3.weight)
        init.zeros_(self.layer3.bias)

    def forward(self, x):
        out1 = F.relu(self.layer1(x))
        out2 = F.relu(self.layer2(out1))
        out3 = self.layer2(out2)

        out = out3 + out1
        out = torch.tanh_(out)

        return out


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(
            torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

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


def orthonorm(Q, eps=1e-7):
    m = torch.tensor(Q.shape[0])  # batch size
    outer_prod = torch.mm(Q.T, Q)
    outer_prod = outer_prod + eps * torch.eye(outer_prod.shape[0]).cuda()

    L = torch.linalg.cholesky(outer_prod)  # lower triangular
    L_inv = torch.linalg.inv(L)
    return torch.sqrt(m) * L_inv.T


class SpectralNet(nn.Module):
    def __init__(self, params):
        super(SpectralNet, self).__init__()
        self.params = params

        input_sz = params['input_dims']
        n_hidden_1 = params['n_hidden_1']
        n_hidden_2 = params['n_hidden_2']
        k = params['num_cluster']

        self.alpha = params['alpha']
        self.fc1 = nn.Linear(input_sz, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_1)
        self.fc3 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc4 = nn.Linear(n_hidden_2, k)
        self.fc5 = nn.Linear(k, k)

        self.A = torch.rand(k, k).cuda()
        self.A.requires_grad = False
        self.cluster_layer = Parameter(torch.Tensor(k, k))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, ortho_step=False):
        self.A.requires_grad = False
        if ortho_step:
            with torch.no_grad():
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                Y_tilde = torch.tanh(self.fc4(x))
                self.A.requires_grad = False

                self.A = orthonorm(Y_tilde, eps=self.params['epsilon'])

                # for debugging
                Y = torch.mm(Y_tilde, self.A)
                res = (1/Y.shape[0]) * torch.mm(Y.T, Y)
                return res

        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            Y_tilde = torch.tanh(self.fc4(x))
            # need to multiply from the right, not from the left
            Y = torch.mm(Y_tilde, self.A)

            q = 1.0 / (1.0 + torch.sum(torch.pow(Y.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            
            return Y,q
