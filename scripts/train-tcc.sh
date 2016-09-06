#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm TCC."
$TIME $BASE/jcp_train.sh -tcc -c 0 -m mnist_576_rbf_8vr.tcc.libsvmmodel $TRAINING_SET 

echo;
echo "Training jlibsvm TCC."
$TIME $BASE/jcp_train.sh -tcc -c 1 -m mnist_576_rbf_8vr.tcc.jlibsvmmodel $TRAINING_SET

echo;
echo "Training jliblinear TCC."
$TIME $BASE/jcp_train.sh -tcc -c 2 -m mnist_576_rbf_8vr.tcc.jliblinearmodel $TRAINING_SET

echo;
echo "Training OpenCV SVM TCC."
$TIME $BASE/jcp_train.sh -tcc -c 3 -m mnist_576_rbf_8vr.tcc.ocvsvmmodel $TRAINING_SET

echo;
echo "Training OpenCV RF TCC."
$TIME $BASE/jcp_train.sh -tcc -c 4 -m mnist_576_rbf_8vr.tcc.ocvrfmodel $TRAINING_SET
