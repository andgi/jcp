#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TRAINING_SET=$DIR/../../pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

export TEST_SET=$DIR/../../pisvm-datasets/classification/mnist_test_576_rbf_8vr.500.dat


$DIR/train-all.sh &> output.txt

echo >> output.txt
echo >> output.txt
echo >> output.txt

$DIR/test-all.sh &>> output.txt
