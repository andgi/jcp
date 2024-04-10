#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TRAINING_SET=$DIR/../../pisvm-datasets/classification/mnist_train_576_rbf_8vr.1000.dat

export TEST_SET=$DIR/../../pisvm-datasets/classification/mnist_test_576_rbf_8vr.5000.dat

export MODELNAME=mnist_576_rbf_8vr

export TRAINING_ARGS="-nc 0"

$DIR/train-all.sh &> output_$MODELNAME.txt

echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt

$DIR/test-all.sh &>> output_$MODELNAME.txt
