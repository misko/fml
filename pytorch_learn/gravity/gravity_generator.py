#!/usr/bin/python

import sys
import numpy as np
import argparse
import scipy.spatial.distance
import cPickle as pickle

G=6.674e-11

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--output', type=str, default="./out.gra")
argparser.add_argument('--examples', type=int, default=10)
argparser.add_argument('--maxbodies', type=int, default=10)
argparser.add_argument('--minbodies', type=int, default=2)
argparser.add_argument('--maxw', type=int, default=1000)
argparser.add_argument('--minw', type=int, default=1)
argparser.add_argument('--boxsize', type=int, default=10)
args = argparser.parse_args()


planets=np.floor(np.random.rand(args.examples)*(args.maxbodies-args.minbodies+1)+args.minbodies).astype(np.int)
xyzs=np.random.rand(planets.sum(),3)*args.boxsize-args.boxsize/2
ms=np.random.rand(planets.sum(),1)*(args.maxw-args.minw)+args.minw


so_far=0
out_e=[]
for x in planets: # the next x planets are interacting
    c_xyzs=xyzs[so_far:so_far+x]
    c_ms=ms[so_far:so_far+x]
    #get the distance between all points
    distances=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(c_xyzs))
    #get a matrix of mass products for all points
    m=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(c_ms, lambda u, v: u*v))*G
    f=np.zeros((x,3))
    E=np.zeros(x)
    ru=np.zeros((x,x,3))
    for i in xrange(x):
        for j in xrange(x):
            if i!=j:
                rv=c_xyzs[i]-c_xyzs[j]
                r=np.linalg.norm(rv)
                ru[i,j,:]=(c_xyzs[i]-c_xyzs[j])/pow(r,3)
                f[i]+=m[i,j]*ru[i,j,:]
                E[i]+=m[i,j]/r
    #atom discriptors are the masses
    #distances 
    out_e.append({'distances':distances,'mass':c_ms,'energy':sum(E),'force':f})
    #out=np.zeros((x,8)) # x, y ,z ,m, E , f_x, f_y, f_z
    #out[:,:3]=c_xyzs #xyz locations
    #out[:,3]=c_ms[:,0] #masses
    #out[:,4]=E #energy
    #out[:,5:]=f #force
    #out_e.append(out)
    so_far+=x

#A.shape = atoms x atom descriptor
#d.shape = atoms x atoms = pair wise distances
#energy = [float] 

pickle.dump(out_e,open(args.output,'wb')) # ? x 8
