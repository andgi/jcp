#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 0 -m $MODELNAME.lctcc.libsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -tcc -c 0 -m $MODELNAME.lctcc.libsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jlibsvm LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 1 -m $MODELNAME.lctcc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -tcc -c 1 -m $MODELNAME.lctcc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jliblinear LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 2 -m $MODELNAME.lctcc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -tcc -c 2 -m $MODELNAME.lctcc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training Neuroph LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 3 -m $MODELNAME.lctcc.neurophmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -tcc -c 3 -m $MODELNAME.lctcc.neurophmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training DeepLearning4j LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 4 -m $MODELNAME.lctcc.deeplearning4jmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -lccc -tcc -c 4 -m $MODELNAME.lctcc.deeplearning4jmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV SVM LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 3 -m $MODELNAME.lctcc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -lccc -tcc -c 3 -m $MODELNAME.lctcc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV RF LCTCC." $BASE/jcp_train.sh -lccc -tcc -c 4 -m $MODELNAME.lctcc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -lccc -tcc -c 4 -m $MODELNAME.lctcc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
