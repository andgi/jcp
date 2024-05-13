#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm LCICC." $BASE/jcp_train.sh -lccc -c 0 -m $MODELNAME.lcicc.libsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -c 0 -m $MODELNAME.lcicc.libsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jlibsvm LCICC." $BASE/jcp_train.sh -lccc -c 1 -m $MODELNAME.lcicc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -c 1 -m $MODELNAME.lcicc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jliblinear LCICC." $BASE/jcp_train.sh -lccc -c 2 -m $MODELNAME.lcicc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -c 2 -m $MODELNAME.lcicc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV SVM LCICC." $BASE/jcp_train.sh -lccc -c 3 -m $MODELNAME.lcicc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -lccc -c 3 -m $MODELNAME.lcicc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV RF LCICC." $BASE/jcp_train.sh -lccc -c 4 -m $MODELNAME.lcicc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -lccc -c 4 -m $MODELNAME.lcicc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
