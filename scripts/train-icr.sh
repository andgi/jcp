#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/libsvm-datasets/regression/cpusmall_scale-training.txt

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the regressors

echo;
echo "Training jlibsvm ICR." $BASE/jcp_train.sh -r 0 -m $MODELNAME.icr.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -r 0 -m $MODELNAME.icr.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jliblinear ICR." $BASE/jcp_train.sh -r 1 -m $MODELNAME.icr.jliblinearmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -r 1 -m $MODELNAME.icr.jliblinearmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training DeepLearning4j ICR." $BASE/jcp_train.sh -r 2 -m $MODELNAME.icr.deeplearning4j $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -r 2 -m $MODELNAME.icr.deeplearning4jmodel $TRAINING_ARGS $TRAINING_SET

