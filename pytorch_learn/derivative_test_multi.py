import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import tempfile

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--markersize', type=int, default=3)
argparser.add_argument('--batchnorm', type=int, default=0)
argparser.add_argument('--nfilters', type=int, default=16)
argparser.add_argument('--nlayers', type=int, default=1)
argparser.add_argument('--inputdim', type=int, default=1)
argparser.add_argument('--lr', type=float, default=3e-3)
argparser.add_argument('--iterations', type=int, default=15000)
argparser.add_argument('--mb', type=int, default=32)
argparser.add_argument('--transfer', type=str, default="relu")
argparser.add_argument('--xmax', type=int, default=10)
argparser.add_argument('--xmin', type=int, default=-10)
argparser.add_argument('--showfig', type=int, default=1)
argparser.add_argument('--df', type=float, default=1.0)
argparser.add_argument('--f', type=float, default=1.0)
argparser.add_argument('--outdir', type=str, default='')
args = argparser.parse_args()

def prefix():
        return 'history_%s_lr%0.4f_nlayers%d_nfilters%d_bn%d_indims%d_df%0.3f_f%0.3f' % (args.transfer,args.lr,args.nlayers,args.nfilters,args.inputdim,args.batchnorm,args.df,args.f)
import os
if args.outdir=='':
    args.outdir=prefix()
try:
    os.makedirs(args.outdir)
except:
    pass
frame=0


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

plt.show(block=False)

last_save_file=""

def save():
    os.system("ffmpeg -r 1 -i "+args.outdir+'/'+prefix()+"_frame%01d.png -vcodec h264 -b:v 5000k  -y "+args.outdir+"/"+prefix()+".mp4")
    os.system("mv "+last_save_file + " " + args.outdir+"/"+prefix()+".png")
    print "mv "+last_save_file + " " + args.outdir+"/"+prefix()+".png"
    for i in xrange(frame-1):
        fn=args.outdir+'/%s_frame%d.png' % (prefix(),i)
        os.system("rm "+fn)

def plot(datas):
        global frame
        global last_save_file
        plt.close()
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(18,8))
        ax1.set_yscale('log')
        ax1.set_title('log')

	colors=["ro","bo","go","mo","ko"]
	keys=datas.keys()
	keys.sort()
	for idx in xrange(len(datas)):
		data=datas[keys[idx]]
		x=list(data[i][0] for i in xrange(len(data)))
		y=list(min(data[i][1],1000) for i in xrange(len(data)))
		#plt.plot(x, y, colors[idx],label=keys[idx],alpha=0.1,markersize=args.markersize)
		ax1.plot(x, y, colors[idx],label=keys[idx],alpha=0.1,markersize=args.markersize)
	ax1.legend(loc='lower left', shadow=True)
        ax1.set_ylim([0.001, 1000+0.001])
	ax1.set_ylabel('err')
	ax1.set_xlabel('iteration')
        pts=1000
        x=np.repeat(np.linspace(float(args.xmin-5)/args.inputdim,float(args.xmax+5)/args.inputdim,pts).reshape(pts,1),args.inputdim,axis=1)
        inp=Variable(torch.from_numpy(x.reshape(pts,args.inputdim)).float())
        y=network(inp).resize(pts)
	ax2.plot(x.sum(1), y.data.numpy(), colors[0],alpha=0.1,markersize=args.markersize,label="network output")
        t=target_f(inp).resize(pts)
	ax2.plot(x.sum(1), t.data.numpy(), colors[1],alpha=0.1,markersize=args.markersize,label="target function")
        ax2.axvline(x=args.xmin,label="training data limit")
        ax2.axvline(x=args.xmax)
        ax2.legend()
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.draw()
        fn=args.outdir+'/%s_frame%d.png' % (prefix(),frame)
	plt.savefig(fn)
        last_save_file=fn
        frame+=1
        #if frame%100==0:
        #    save()
        if args.showfig==1:
	    plt.pause(0.001)

def make_sin(n):
    #xs = (args.xmax-args.xmin)*np.random.rand(n,args.inputdim)/args.inputdim+args.xmin
    xs = ((args.xmax-args.xmin)*np.random.uniform(size=(n,args.inputdim))+args.xmin)/args.inputdim
    zs = xs.sum(1)
    #zs = (xs*np.linspace(1,args.inputdim,args.inputdim)).sum(1)
    ys = np.sin(zs)
    #df = np.repeat(np.cos(zs).reshape(n,1),args.inputdim,axis=1)*np.linspace(1,args.inputdim,args.inputdim)
    df = np.repeat(np.cos(zs).reshape(n,1),args.inputdim,axis=1)
    return {'xs':xs,'ys':ys,'df':df,'sz':n}

class TargetFunctional(torch.nn.Module):
    def __init__(self):
        super(TargetFunctional, self).__init__()

    def forward(self, input):
        x=0.4*args.inputdim*(input+input.cos())
        #x=x*x+x*(x.pow(2)).cos()
        x=x+x*(x.pow(2)).cos()
        x=x.sum(1)
        return x#+input.cos()

target_f = TargetFunctional()
def make_multi(n):
    xs = ((args.xmax-args.xmin)*np.random.uniform(size=(n,args.inputdim))+args.xmin)/args.inputdim
    xs_v = Variable(torch.from_numpy(xs).float(),requires_grad=True)
    ys_v = target_f(xs_v)
    ys = ys_v.data.numpy()
    ys_v.sum().backward()
    df_v = xs_v.grad
    df = df_v.data.numpy()
    return {'xs':xs,'ys':ys,'df':df,'sz':n}

#ds_gen=make_sin
ds_gen=make_multi
training_data=ds_gen(100000)
testing_data=ds_gen(100000)

import math
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

network = FK(args.inputdim,args.nfilters)

params=[{'params':network.parameters()}]
#optimizer = torch.optim.SGD(params,1e-7)
optimizer = torch.optim.Adam(params,args.lr)

def get_mb(sz,d):
	idxs=np.floor(np.random.uniform(size=sz)*d['sz']).astype(int)
	xs=d['xs'][idxs].reshape((args.mb,args.inputdim))
	xs=Variable(torch.from_numpy(xs).float(),requires_grad = True) #.unsqueeze(1)
	#xs=Variable(torch.from_numpy(xs).float())
	ys=d['ys'][idxs].reshape((args.mb,1))
	ys=Variable(torch.from_numpy(ys).float()) #.unsqueeze(1)
	df=d['df'][idxs]
	df=Variable(torch.from_numpy(df).float()) #.unsqueeze(1)
	return xs,ys,df

f_training_history=[]
f_testing_history=[]	
df_training_history=[]
df_testing_history=[]	
mean_history=[]

print(torch.__version__)

#optimization iterations
for iteration in range(args.iterations):
	training=(iteration%10!=0)
	data=training_data
	s="TRAIN"
	if not training:
		s="TEST"
		data=testing_data
	network.train(training)
	#get a minibatch
	xs,ys,df=get_mb(args.mb,data)
	#run the network
	network_output=network(xs)

	optimizer.zero_grad()

        #lets get the derivative
        f_loss = (network_output-ys).pow(2).mean()
	(args.f*f_loss).backward(retain_graph=True,create_graph=True)

        #erase the weight updates from the err function
        #optimizer.zero_grad()

        #get the gradient pentaly
        df_loss=(df-torch.autograd.grad([ network_output[x] for x in xrange(args.mb) ],xs,create_graph=True)[0].resize(args.mb,args.inputdim)).pow(2).mean()
        (args.df*df_loss).backward()

        #print params[0]['params'][0].grad.sum()
	mean_history.append((iteration,ys.pow(2).sum().data[0]/args.mb))
	if training:
		optimizer.step()
		f_training_history.append((iteration,f_loss.data[0]))
		df_training_history.append((iteration,df_loss.data[0]))
	elif iteration%100==0:
		f_testing_history.append((iteration,f_loss.data[0]))
		df_testing_history.append((iteration,df_loss.data[0]))
                plot({"f_training":f_training_history,"f_testing":f_testing_history,"mean":mean_history,"df_training":df_training_history,"df_testing":df_testing_history})
	if args.showfig==1:
		print iteration,s,f_loss.data[0],ys.pow(2).sum().data[0]/args.mb
	
save()
