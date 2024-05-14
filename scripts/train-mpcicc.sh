#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm MPC ICC." $BASE/jcp_train.sh -mpc -c 0 -m $MODELNAME.mpcicc.libsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -c 0 -m $MODELNAME.mpcicc.libsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jlibsvm MPC ICC." $BASE/jcp_train.sh -mpc -c 1 -m $MODELNAME.mpcicc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -c 1 -m $MODELNAME.mpcicc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jliblinear MPC ICC." $BASE/jcp_train.sh -mpc -c 2 -m $MODELNAME.mpcicc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -c 2 -m $MODELNAME.mpcicc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training Neuroph MPC ICC." $BASE/jcp_train.sh -mpc -c 3 -m $MODELNAME.mpcicc.neurophmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -c 3 -m $MODELNAME.mpcicc.neurophmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV SVM MPC ICC." $BASE/jcp_train.sh -c 3 -mpc -m $MODELNAME.mpcicc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -c 3 -mpc -m $MODELNAME.mpcicc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV RF MPC ICC." $BASE/jcp_train.sh -c 4 -mpc -m $MODELNAME.mpcicc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -c 4 -mpc -m $MODELNAME.mpcicc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
