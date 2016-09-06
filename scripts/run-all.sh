#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TRAINING_SET=$DIR/../../pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

export TEST_SET=$DIR/../../pisvm-datasets/classification/mnist_test_576_rbf_8vr.500.dat


$DIR/train-icc.sh &> output.txt
echo >> output.txt
echo >> output.txt
$DIR/train-lcicc.sh &>> output.txt
echo >> output.txt
echo >> output.txt
$DIR/train-tcc.sh &>> output.txt
echo >> output.txt
echo >> output.txt
$DIR/train-lctcc.sh &>> output.txt

echo >> output.txt
echo >> output.txt
echo >> output.txt

$DIR/test-icc.sh &>> output.txt
echo >> output.txt
echo >> output.txt
$DIR/test-lcicc.sh &>> output.txt
echo >> output.txt
echo >> output.txt
$DIR/test-tcc.sh &>> output.txt
echo >> output.txt
echo >> output.txt
$DIR/test-lctcc.sh &>> output.txt
