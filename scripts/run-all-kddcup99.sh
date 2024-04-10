#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TRAINING_SET=$DIR/../../pisvm-datasets/classification/kddcup99_nvr_train.scale.100000.dat

export TEST_SET=$DIR/../../pisvm-datasets/classification/kddcup99_nvr_test.scale.5000.dat

export MODELNAME=kddcup99_nvr

export TRAINING_ARGS="-nc 0"

$DIR/train-all.sh &> output_$MODELNAME.txt

echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt
echo >> output_$MODELNAME.txt

$DIR/test-all.sh &>> output_$MODELNAME.txt
