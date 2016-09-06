#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm LCICC."
$TIME $BASE/jcp_train.sh -lccc -c 0 -m mnist_576_rbf_8vr.lcicc.libsvmmodel $TRAINING_SET 

echo;
echo "Training jlibsvm LCICC."
$TIME $BASE/jcp_train.sh -lccc -c 1 -m mnist_576_rbf_8vr.lcicc.jlibsvmmodel $TRAINING_SET

echo;
echo "Training jliblinear LCICC."
$TIME $BASE/jcp_train.sh -lccc -c 2 -m mnist_576_rbf_8vr.lcicc.jliblinearmodel $TRAINING_SET

echo;
echo "Training OpenCV SVM LCICC."
$TIME $BASE/jcp_train.sh -lccc -c 3 -m mnist_576_rbf_8vr.lcicc.ocvsvmmodel $TRAINING_SET

echo;
echo "Training OpenCV RF LCICC."
$TIME $BASE/jcp_train.sh -lccc -c 4 -m mnist_576_rbf_8vr.lcicc.ocvrfmodel $TRAINING_SET
