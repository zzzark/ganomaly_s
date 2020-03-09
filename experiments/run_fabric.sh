#!/bin/bash

# Run fabric experiment for each individual dataset.
# For each anomalous digit

for i in {1024,2048}
do
    for m in {3..5}
    do
        echo "Running Fabric  ###############"
	echo "Manual Seed: $m ###############"
        python train.py --dataset fabric --isize 128 --nc 3 --niter 100 --batchsize 32 --nz $i --manualseed $m --display --strengthen --lr 0.00005
    done
done
exit 0
