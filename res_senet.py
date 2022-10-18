import torch
import torch.nn as nn

class mlp(nn.Module):
	def __init__(self, input_dims, hid_dims, out_dims):
		super(mlp, self).__init__()
		self.input_dims = input_dims
		self.hid_dims = hid_dims
		self.output_dims = out_dims
		self.layer = nn.Linear(self.input_dims,self.output_dims)


	def forward(self,x):
		out = nn.ReLU(self.layer(x))

		return out