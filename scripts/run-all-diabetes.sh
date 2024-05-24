#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TRAINING_SET=$DIR/../../libsvm-datasets/2-class/diabetes_scale-training.txt

export TEST_SET=$DIR/../../libsvm-datasets/2-class/diabetes_scale-test.txt

export MODELNAME=diabetes

export TRAINING_ARGS="-nc 0"

$DIR/train-all.sh &> output_$MODELNAME.txt

echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt

$DIR/test-all.sh &>> output_$MODELNAME.txt
