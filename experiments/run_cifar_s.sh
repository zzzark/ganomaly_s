#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("plane" "car" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
for i in "${arr[@]}";
do
    for m in {0..2}
    do
	echo "Manual Seed: $m"
        echo "Running CIFAR. Anomaly Class: $i "
        python train.py --dataset cifar10 --isize 32 --niter 15 --nz 256 --manualseed $m --abnormal_class $i
	python train_z.py --dataset cifar10 --isize 32 --load_weights --strengthen 1 --display --nz 256 --classifier --nc 3 --z_metric roc --manualseed $m --abnormal_class $i
    done
done
exit 0
