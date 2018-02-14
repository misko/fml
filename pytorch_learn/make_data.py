import random
import sys


if len(sys.argv)!=1:
	print "NO ARGS!" 
	sys.exit(1)

atom_types=2
box=10
atoms=int(8*random.random())+2

for i in range(atoms):
	x=box*random.random()
	y=box*random.random()
	z=box*random.random()
	t=int(atom_types*random.random())
	print x,y,z,t

print random.random()*100
