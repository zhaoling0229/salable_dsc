import torch

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def compute_se_loss(coef,feature):
    pass

def compute_spec_loss(p,y):
    pass