#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm LCTCC."
$TIME $BASE/jcp_train.sh -lccc -tcc -c 0 -m mnist_576_rbf_8vr.lctcc.libsvmmodel $TRAINING_SET 

echo;
echo "Training jlibsvm LCTCC."
$TIME $BASE/jcp_train.sh -lccc -tcc -c 1 -m mnist_576_rbf_8vr.lctcc.jlibsvmmodel $TRAINING_SET

echo;
echo "Training jliblinear LCTCC."
$TIME $BASE/jcp_train.sh -lccc -tcc -c 2 -m mnist_576_rbf_8vr.lctcc.jliblinearmodel $TRAINING_SET

echo;
echo "Training OpenCV SVM LCTCC."
$TIME $BASE/jcp_train.sh -lccc -tcc -c 3 -m mnist_576_rbf_8vr.lctcc.ocvsvmmodel $TRAINING_SET

echo;
echo "Training OpenCV RF LCTCC."
$TIME $BASE/jcp_train.sh -lccc -tcc -c 4 -m mnist_576_rbf_8vr.lctcc.ocvrfmodel $TRAINING_SET
