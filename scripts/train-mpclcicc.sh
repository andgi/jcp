#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm MPC LCICC." $BASE/jcp_train.sh -mpc -lccc -c 0 -m mnist_576_rbf_8vr.mpclcicc.libsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -lccc -c 0 -m mnist_576_rbf_8vr.mpclcicc.libsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jlibsvm MPC LCICC." $BASE/jcp_train.sh -mpc -lccc -c 1 -m mnist_576_rbf_8vr.mpclcicc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -lccc -c 1 -m mnist_576_rbf_8vr.mpclcicc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jliblinear MPC LCICC." $BASE/jcp_train.sh -mpc -lccc -c 2 -m mnist_576_rbf_8vr.mpclcicc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -lccc -c 2 -m mnist_576_rbf_8vr.mpclcicc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training OpenCV SVM MPC LCICC." $BASE/jcp_train.sh -mpc -lccc -c 3 -m mnist_576_rbf_8vr.mpclcicc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -lccc -c 3 -m mnist_576_rbf_8vr.mpclcicc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training OpenCV RF MPC LCICC." $BASE/jcp_train.sh -mpc -lccc -c 4 -m mnist_576_rbf_8vr.mpclcicc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -mpc -lccc -c 4 -m mnist_576_rbf_8vr.mpclcicc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
