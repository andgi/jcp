#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TRAINING_SET=$DIR/../../libsvm-datasets/regression/cpusmall_scale-training.txt

export TEST_SET=$DIR/../../libsvm-datasets/regression/cpusmall_scale-test.txt

export MODELNAME=cpusmall

export TRAINING_ARGS="-nc 0"

$DIR/train-icr.sh &> output_$MODELNAME.txt

echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt

$DIR/test-icr.sh &>> output_$MODELNAME.txt
