#!/bin/bash

max_runs=12
runs=0

lrs="0.001 0.0001"
inputdims="1 2 4"
transfers="relu four selu taylor"
layers="2 4 16"
filters="8 16 64"
batchnorms="0 1"
dfs="0 1"
fs="0 1"
for bn in $batchnorms; do
for lr in $lrs; do
for i in $inputdims; do
for nf in $filters; do 
for t in $transfers; do
for df in $dfs; do
for f in $fs; do
for layer in $layers; do
	if [ "$t" == "taylor" -a $nf -gt 20 ]; then
		continue
	fi
	if [ $f -eq 0 -a $df -eq 0 ] ; then
		continue
	fi
	python derivative_test_multi.py --inputdim $i --lr $lr --nlayers $layer --transfer $t --batchnorm $bn --df $df --f $f --showfig 0 &
	#python atomblaster.py --markersize 4 --distpower $distpower --nfilters $filter --nlayers 1 --lr $lr --radius $radius --includeh $includeh save.data &
	#sleep 10 &
	runs=`expr ${runs} + 1`
	while [ "${runs}" -ge "${max_runs}" ]; do
		wait
		runs=0
	done
done
done
done
done
done
done
done
done

exit


#check if distpower 0/1 matters
python atomblaster.py --markersize 4 --distpower 0 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 2 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 3 --nfilters 4 --nlayers 1 --lr 0.001 --radius 3 --includeh 1 save.data &
wait
wait
wait 
wait
#check if distpower includeh matters
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 9 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 9 --includeh 0 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 6 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 1 --lr 0.001 --radius 12 --includeh 1 save.data &
wait
wait
wait
wait
#check if distpower filters matters
python atomblaster.py --markersize 4 --distpower 1 --nfilters 8 --nlayers 1 --lr 0.001 --radius 9 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 16 --nlayers 1 --lr 0.001 --radius 9 --includeh 1 save.data &
python atomblaster.py --markersize 4 --distpower 1 --nfilters 4 --nlayers 2 --lr 0.001 --radius 9 --includeh 1 save.data &
wait
wait
wait

