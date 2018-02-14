import torch
from torch.autograd import Variable

vp=Variable(torch.zeros(1))
v=Variable(torch.zeros(1))
print v.sum()
print v.sum()
vp[0]+=v.sum()
