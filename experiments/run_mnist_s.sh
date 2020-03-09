#!/bin/bash

# Run MNIST experiment for each individual dataset.
# For each anomalous digit
for i in {0..9}
do
    for m in {0..2}
    do
	echo "#Manual Seed: $m"
        echo "#Running MNIST2, Abnormal Digit: $i"
        python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class $i --manualseed $m --proportion 0.2 --lr 0.002 --display --strengthen --beta1 0.5
    done
done
exit 0
