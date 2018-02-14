import torch


a=torch.FloatTensor(5,7)
print(a)
print(a.size())

from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)  # notice the "Variable containing" line

print(Variable(torch.LongTensor([3])))
