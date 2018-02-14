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
argparser.add_argument('path', nargs='+', help='Path of a file or a folder of files.')
argparser.add_argument('--markersize', type=int, default=3)
argparser.add_argument('--distpower', type=int, default=1)
argparser.add_argument('--batchnorm', type=int, default=0)
argparser.add_argument('--nfilters', type=int, default=4)
argparser.add_argument('--nlayers', type=int, default=1)
argparser.add_argument('--lr', type=float, default=3e-3)
argparser.add_argument('--radius', type=float, default=9)
argparser.add_argument('--iterations', type=int, default=15000)
argparser.add_argument('--includeh', type=int, default=1)
argparser.add_argument('--mb', type=int, default=32)
args = argparser.parse_args()

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

plt.show(block=False)

mx=50


def plot(datas):
	plt.clf()
	plt.yscale('log')
	plt.title('log')
	colors=["ro","bo","go"]
	keys=datas.keys()
	keys.sort()
	for idx in xrange(len(datas)):
		data=datas[keys[idx]]
		x=list(data[i][0] for i in xrange(len(data)))
		y=list(min(data[i][1],mx) for i in xrange(len(data)))
		plt.plot(x, y, colors[idx],label=keys[idx],alpha=0.1,markersize=args.markersize)
	plt.legend(loc='lower left', shadow=True)
	plt.ylim(0.001, mx+0.001)
	plt.ylabel('err')
	plt.xlabel('iteration')
	plt.draw()
	plt.savefig('history_relu_lr%0.4f_nlayers%d_nfilters%d_bn%d_radius%d_distpower%d_includeh%d_zerofills.png' % (args.lr,args.nlayers,args.nfilters,args.batchnorm,args.radius,args.distpower,args.includeh))
	plt.pause(0.001)

atom_types={}

def xyz_to_d(A):
	D = np.zeros((A.shape[0],A.shape[0]))
	for x in range(A.shape[0]):
		for y in range(A.shape[0]):
			d = A[x][:3]-A[y][:3]
			d = np.sqrt(np.dot(d,d))
			D[x,y]=d
			D[y,x]=d
	return D	

def int_to_bitv(b,sz):
	bv=np.zeros((sz))
	bv[b]=1
	return bv


def distances_to_neighbours(d,r):
	neighbours=[]
	for i in xrange(len(d)):
		i_neighbours=[]
		for j in xrange(len(d)):
			if d[i][j]<=r:
				i_neighbours.append(j)
		neighbours.append(i_neighbours)
	#neighbours=np.array(neighbours)
	return neighbours

def read_qm9_xyz(fn):
        print "READING",fn
	f=open(fn)
	atoms=[]
	A=[]
	lines=f.readlines()
	natoms=int(lines[0].strip())
	energy=float(lines[1].strip().split()[15])
	for atom in lines[2:2+natoms]:
		atom=atom.strip().replace('*^','e').split()
		atoms.append((float(atom[1]),float(atom[2]),float(atom[3]),float(atom[4])))
		A.append(int_to_bitv(atom_types[atom[0]],len(atom_types)))
	atoms=np.array(atoms)
	d=xyz_to_d(atoms)
	A=np.array(A)
	#neighbours=distances_to_neighbours(d)
	return A,d,energy

def read_xyz(fn):
	f=open(fn)
	atoms=[]
	energy=0
	A=[]
	for line in f:
		line=line.split()
		if len(line)==4:
			atoms.append((float(line[0]),float(line[1]),float(line[2]),int(line[3])))
			A.append(int_to_bitv(int(line[3]),atom_types))
		else:
			energy=float(line[0])
	atoms=np.array(atoms)
	d=xyz_to_d(atoms)
	A=np.array(A)
	return A,d,energy


import cPickle as pickle

training_data={'atoms':[],'energies':[],'distances':[],'sz':0,'neighbours':[],'forces':[]}
testing_data={'atoms':[],'energies':[],'distances':[],'sz':0,'neighbours':[],'forces':[]}


mode=""

if len(args.path)==1 and args.path[0][-5:]=='.data':
	print "LOAD DATA"
	d=pickle.load(open(args.path[0],"rb"))
	training_data=d['training_data']
	testing_data=d['testing_data']
        print "Loaded",len(training_data),"training",len(testing_data),'testing','examples'
        mode=d['mode']
        atom_types=d['atom_types']
	print "LOAD DATA - DONE"
	ds=[training_data, testing_data]
	for d in ds:
		for i in xrange(len(d['atoms'])):
			d['atoms'][i]=Variable(torch.FloatTensor(d['atoms'][i])).float()
		d['neighbours']=[]
		for i in xrange(len(d['distances'])):
			d['neighbours'].append(distances_to_neighbours(d['distances'][i],args.radius))
			d['distances'][i]=Variable(torch.FloatTensor(d['distances'][i])).float()
		if args.includeh==0:
			for i in xrange(len(d['atoms'])):
				natoms=d['atoms'][i].size()[0]
				for j in xrange(natoms):
					if d['atoms'][i][j][atom_types['H']].data[0]>0.0001:
						d['neighbours'][i][j]=[]
	
else:
	print "PROCESS DATA"
        if len(args.path)==1 and args.path[0][-5:]==".list":
		f=open(args.path[0])
		args.path=[]
		for line in f:
			args.path.append(line.strip())
	#construct the dataset
	idx=0
	for fn in args.path:
		idx+=1
		if idx%1000==0:
			print idx
                if fn.split('.')[-1]=='xyz':
                    if mode=="":
                        print "QM9 mode"
                        mode="qm9"
	                atom_types={'H':0,'C':1,'O':2,'N':3,'F':4}
                    if mode!="qm9":
                        print "whoops cant have two modes"
                        sys.exit(1)
                    A,d,energy=read_qm9_xyz(fn)
                    #A.shape = atoms x atom descriptor
                    #d.shape = atoms x atoms = pair wise distances
                    #energy = [float] 
                    data=training_data
                    if np.random.rand(1)[0]>0.7:
                            data=testing_data
                    data['atoms'].append(A) 
                    data['energies'].append(energy)
                    data['distances'].append(d)
                    data['sz']+=1
                elif fn.split('.')[-1]=='gra':
                    if mode=="":
                        print "Gravity mode"
                        mode="gravity"
                        atom_types={"mass":0}
                    if mode!="gravity":
                        print "whoops cant have two modes"
                        sys.exit(1)
                    full_data=pickle.load(open(fn,"rb"))
                    for d in full_data:
                        data=training_data
                        if np.random.rand(1)[0]>0.7:
                                data=testing_data
                        data['atoms'].append(d['mass'])
                        data['energies'].append(d['energy'])
                        data['distances'].append(d['distances'])
                        data['forces'].append(d['force'])
                        data['sz']+=1
                    
	training_data['energies']=np.array(training_data['energies'])
	testing_data['energies']=np.array(testing_data['energies'])
	u=training_data['energies'].mean()
	o=training_data['energies'].std()
	testing_data['energies']-=u
	testing_data['energies']/=o
	training_data['energies']-=u
	training_data['energies']/=o

        pickle.dump({'training_data':training_data,'testing_data':testing_data,'mode':mode,'atom_types':atom_types},open('save.data','wb'))
	print "SAVED TO save.data",mode
	sys.exit(1)

#The basic layer 
class FK(torch.nn.Module):
    def __init__(self,f,h):
        super(FK, self).__init__()
        self.fc1 = nn.Linear(f, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, 1)
	if args.batchnorm==1:
		self.bn1 = nn.BatchNorm1d(h,affine=True)
		self.bn2 = nn.BatchNorm1d(h,affine=True)
		self.bn3 = nn.BatchNorm1d(h)

    def forward(self, input):
	nl = F.relu
	#nl = selu
        x = nl(self.fc1(input))
	if args.batchnorm==1:
		x = self.bn1(x)
        x = nl(self.fc2(x))
	if args.batchnorm==1:
		x = self.bn2(x)
        x = nl(self.fc3(x))
	if args.batchnorm==1:
		x = self.bn3(x)
        x = self.fc4(x)
        return x

params=[]

#build filter bank for layer 1
fss=[]
fs1=[]
for i in range(args.nfilters):
	fs1.append(FK(len(atom_types)*2+1,args.nfilters*2))
	params.append({'params':fs1[-1].parameters()})
fss.append(fs1)
for n in xrange(args.nlayers-1):
	fs=[]
	for i in range(args.nfilters):
		fs.append(FK(args.nfilters*2+1,args.nfilters*2))
		params.append({'params':fs[-1].parameters()})
	fss.append(fs)

#optimizer = torch.optim.SGD(params,1e-7)
optimizer = torch.optim.Adam(params,args.lr)

def get_mb(sz,d):
	idxs=np.floor(np.random.rand(sz)*d['sz']).astype(int)
	energies=d['energies'][idxs]
	energies=Variable(torch.from_numpy(energies).float()) #.unsqueeze(1)
	atoms=list(d['atoms'][i] for i in idxs)
	distances=list(d['distances'][i] for i in idxs)
	neighbours=list(d['neighbours'][i] for i in idxs)
	return atoms,distances,neighbours,energies

def run_layer(atoms,distances,neighbours,fs):
	#prepare the input tensor
	vs=[]
	for mb_idx in xrange(args.mb): # for each example
		natoms=distances[mb_idx].size()[0]
		for atom_i in range(natoms): # for each atom i
			for atom_j in neighbours[mb_idx][atom_i]: # for each atom j in neighbourhod of i
				if args.distpower>=0:
					vs.append(torch.cat([atoms[mb_idx].select(0,atom_i),atoms[mb_idx].select(0,atom_j),distances[mb_idx][atom_i,atom_j].pow(args.distpower)]).unsqueeze(1).transpose(1,0)) # run both of them through 
				else:
					vs.append(torch.cat([atoms[mb_idx].select(0,atom_i),atoms[mb_idx].select(0,atom_j),1.0/distances[mb_idx][atom_i,atom_j].pow(-args.distpower)]).unsqueeze(1).transpose(1,0)) # run both of them through 
				#vs.append(torch.cat([atoms[mb_idx].select(0,atom_i),atoms[mb_idx].select(0,atom_j)]).unsqueeze(1).transpose(1,0))
	vs=torch.cat(vs,0)
	#run each filter
	fnvs=[]
	for fn in xrange(args.nfilters):
		fnvs.append(fs[fn](vs))
	#concatenate to a big tensor
	fnvs=torch.cat(fnvs,1)

	#now create the output tensor
	zero_fill=Variable(torch.zeros(args.nfilters,1)) # if an atom has no neighbours, still need to pass up values to next layer
	output=[]
	idx=0
	for mb_idx in xrange(args.mb): # for each example
		natoms=distances[mb_idx].size()[0]
		vps=[] # the output vector ?
		for atom_i in range(natoms): # for each atom
			if len(neighbours[mb_idx][atom_i])>0:
				vps.append(fnvs.narrow(0,idx,len(neighbours[mb_idx][atom_i])).sum(0).view(-1,1)) #.transpose(1,0))
				idx+=len(neighbours[mb_idx][atom_i])
			else:
				vps.append(zero_fill)
		output.append(torch.cat(vps,1).transpose(1,0))
	return output

def network_forward(atoms,distances,neighbours,targets):
	#prepare the input tensor
	output=run_layer(atoms,distances,neighbours,fss[0])
	#layer2
	for x in xrange(args.nlayers-1):
		output=run_layer(output,distances,neighbours,fss[1+x])
	#collapse everything
	network_output=[]
	for idx in xrange(args.mb):
		new_v=output[idx]
		network_output.append(new_v.sum())
	network_output=torch.cat(network_output,0) #.unsqueeze(1)
	return network_output


training_history=[]
testing_history=[]	
mean_history=[]

#optimization iterations
for iteration in range(args.iterations):
	training=(iteration%10!=0)
	data=training_data
	s="TRAIN"
	if not training:
		s="TEST"
		data=testing_data
	for fs in fss:
		for fn in xrange(args.nfilters):
			fs[fn].train(training)
	#get a minibatch
	atoms,distances,neighbours,targets=get_mb(args.mb,data)
	#run the network
	network_output=network_forward(atoms,distances,neighbours,targets)

	loss_fn = nn.MSELoss() 
	err = loss_fn(network_output, targets)
	optimizer.zero_grad()
	mean_history.append((iteration,targets.pow(2).sum().data[0]/args.mb))
	if training:
		err.backward()
		optimizer.step()
		training_history.append((iteration,err.data[0]))
	elif iteration%100==0:
		print torch.cat([network_output.unsqueeze(1),targets.unsqueeze(1)],1)
		testing_history.append((iteration,err.data[0]))
		plot({"training":training_history,"testing":testing_history,"mean":mean_history})
	print iteration,s,err.data[0],targets.pow(2).sum().data[0]/args.mb
	#if str(err.data[0])=='nan':
	#	break
	
	
