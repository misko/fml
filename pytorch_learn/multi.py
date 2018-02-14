import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--markersize', type=int, default=3)
argparser.add_argument('--nfilters', type=int, default=64)
argparser.add_argument('--nlayers', type=int, default=5)
argparser.add_argument('--inputdim', type=int, default=1)
argparser.add_argument('--lr', type=float, default=3e-3)
argparser.add_argument('--mb', type=int, default=128)
argparser.add_argument('--xmax', type=int, default=10)
argparser.add_argument('--xmin', type=int, default=-10)
argparser.add_argument('--batchnorm', type=int, default=0)
argparser.add_argument('--transfer', type=str, default='relu')
args = argparser.parse_args()



plt.show(block=False)

def plot():
	colors=["ro","bo","go","mo","ko"]
        pts=5000
        plt.close()
        f, ax1 = plt.subplots(1, 1, sharey=False, figsize=(7,4))
        x=np.repeat(np.linspace(float(args.xmin-15)/args.inputdim,float(args.xmax+15)/args.inputdim,pts).reshape(pts,1),args.inputdim,axis=1)
        inp=Variable(torch.from_numpy(x.reshape(pts,args.inputdim)).float())
        #y=netfwd(inp).resize(pts)
        y=network(inp).resize(pts)
	ax1.plot(x.sum(1), y.data.numpy(), colors[0],alpha=0.1,markersize=args.markersize)
        t=target_f(inp).resize(pts)
        #print np.hstack((x,t.data.numpy().reshape(-1,1),y.data.numpy().reshape(-1,1)))
	ax1.plot(x.sum(1), t.data.numpy(), colors[1],alpha=0.1,markersize=args.markersize)
        ax1.axvline(x=args.xmin)
        ax1.axvline(x=args.xmax)
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.draw()
	plt.pause(0.001)


dtype = torch.FloatTensor
def new_w(r,c):
    return Variable(torch.randn(r,c).type(dtype), requires_grad=True)


class four(torch.nn.Module):
    def __init__(self,n,nfilters_in,nfilters_out):
        super(four, self).__init__()
        self.fc = nn.Linear(2*n*nfilters_in,nfilters_out)
        self.n = n

    def forward(self,input):
        xs=[]
        for i in xrange(self.n):
            xs.append((input*i).sin())
            xs.append((input*i).cos())
        xs=torch.cat(xs,1)/self.n
        return self.fc(xs)

import math
class taylor(torch.nn.Module):
    def __init__(self,n,nfilters_in,nfilters_out):
        super(taylor, self).__init__()
        self.fc = nn.Linear(n*nfilters_in,nfilters_out)
        self.n = n
        self.coeffs = [ math.factorial(x) for x in xrange(self.n)]

    def forward(self,input):
        xs=[]
        for i in xrange(self.n):
            xs.append(input.pow(i).clamp(max=100).clamp(min=-100)/self.coeffs[i])
        xs=torch.cat(xs,1)
        return self.fc(xs)
        
def ident(x):
    return x

#The basic layer 
class FK(torch.nn.Module):
    def __init__(self,f,h):
        super(FK, self).__init__()
        self.fc_first = nn.Linear(f, h)
        self.fs = nn.ModuleList() 
        self.bns = nn.ModuleList() 
        for x in xrange(args.nlayers):
            if args.transfer=='taylor':
                self.fs.append(taylor(args.nfilters,args.nfilters,args.nfilters))
            elif args.transfer=='four':
                self.fs.append(four(args.nfilters,args.nfilters,args.nfilters))
            else:
                self.fs.append(nn.Linear(h,h))
	    if args.batchnorm==1:
	        self.bns.append(nn.BatchNorm1d(h,affine=True))
        self.fc_last = nn.Linear(h, 1)

    def forward(self, input):
	nl = F.relu
        if args.transfer=="relu":
            pass
        elif args.transfer=='selu':
            nl = selu
        elif args.transfer in ('four','taylor'):
            nl= ident
        else:
            print "unknown transfer function"
            sys.exit(1)
        x = nl(self.fc_first(input))
        for i in xrange(args.nlayers):
            x = nl(self.fs[i](x))
            if args.batchnorm==1:
                x = self.bns[i](x)
        x = self.fc_last(x)
        return x

layers=[]
def netfwd(inp):
    if len(layers)==0:
        layers.append(([new_w(inp.size()[1],args.nfilters),new_w(1,args.nfilters)],'relu'))
        for n in xrange(args.nlayers-2):
            layers.append(([new_w(args.nfilters,args.nfilters),new_w(1,args.nfilters)],'relu'))
        layers.append(([new_w(args.nfilters,1),new_w(1,1)],'none'))
    x=inp
    for w,t in layers:
        x=x.mm(w[0])
        x+=w[1].expand_as(x)
        if t=='tsin':
            x=x.tanh()+x.sin()
        elif t=='sin':
            x=x.sin()
        elif t=='relu':
            x=x.clamp(min=0)
        elif t=='elu':
            x=x.exp()
        elif t=='prelu':
            m=torch.nn.PReLU()
            x=m(x)
        elif t=='none':
            pass
        else:
            print "WROG TYP"
            sys.exit(1)
    return x

class TargetFunctional(torch.nn.Module):
    def __init__(self):
        super(TargetFunctional, self).__init__()

    def forward(self, input):
        #x=(input).sin()
        #x=((input+8)/5).exp()
        x=input*1.5
        x=x.sin()*x+(x*10).cos()
        #x=x.sum(1)
        #x=x.exp()
        return x

#x=Variable(torch.Tensor(1).uniform_(3,3).type(dtype),requires_grad=True)
#x=Variable(torch.linspace(1,3,3).type(dtype),requires_grad=True)
#y=x*x
#dx = 2x, ddx= 2

#df=torch.autograd.grad([ network_output[x] for x in xrange(args.mb) ],xs,create_graph=True)[0].resize(args.mb,args.inputdim)).pow(2).mean(
#df=torch.autograd.grad([y[i] for i in xrange(3)],x,create_graph=True)[0]
#print df
#dff=torch.autograd.grad([df[i] for i in xrange(3)],x,create_graph=True)[0]
#print df,dff
#sys.exit(1)


def derivatives(y,x,n):
    dfs=[]
    for i in xrange(n):
        if i==0:
            dfs.append(y)
        elif i==1:
            dfs.append(torch.autograd.grad([y[ii] for ii in xrange(y.size()[0])],x,create_graph=True)[0])
        else:
            dfs.append(torch.autograd.grad([dfs[-1][ii] for ii in xrange(y.size()[0])],x,create_graph=True)[0])
    return torch.cat(dfs,1)


network = FK(1,args.nfilters)
target_f = TargetFunctional()
optimizer = None 

nds=3
for i in xrange(10000):
    x=Variable(torch.Tensor(args.mb,args.inputdim).uniform_(args.xmin,args.xmax).type(dtype), requires_grad=True)
    #x=Variable(torch.from_numpy(np.vstack([np.linspace(1,2,100),np.linspace(1,2,100)]).T).float(),requires_grad=True)
    t=target_f(x)
    dfs_true=derivatives(t,x,nds)
    #y.sum().backward()

    #y=netfwd(x)
    y=network(x)
    dfs=derivatives(y,x,nds)

    
    if optimizer==None:
        #params=[]
        #for j in xrange(args.nlayers):
        #    params.append({'params':layers[j][0]})
        params=[{'params':network.parameters()}]
        optimizer=torch.optim.Adam(params, lr=args.lr)

    randomd_d=False
    if random_d:
        idx=int(torch.Tensor(1).uniform_(0,nds).floor()[0])
        dfs=dfs.narrow(1,idx,1)
        dfs_true=dfs_true.narrow(1,idx,1)

    #loss=(t-y).pow(2).mean()
    loss=(dfs-dfs_true).pow(2).clamp(max=10).clamp(min=-10).mean() #add on taylor here? 
    #l1_crit = nn.L1Loss(size_average=False)
    #reg_loss = 0
    #for param in network.parameters():
    #    reg_loss += param.abs().sum() #l1_crit(param)
    #    factor = 0.05
    #    loss += factor * reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print params[0]['params'][0].grad.data.sum(),params[0]['params'][0].data.sum()
    print loss.data[0],t.mean().data[0]
    #print params[1].grad
    if i%10==0:
        plot()
