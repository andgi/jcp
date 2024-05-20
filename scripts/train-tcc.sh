#!/bin/bash

# TRAINING_SET is expected to be exported before invoking this script.
#TRAINING_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_train_576_rbf_8vr.500.dat

# The time program gives better info than the builtin time command.
TIME=`which time`

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# Train the classifiers

echo;
echo "Training libsvm TCC." $BASE/jcp_train.sh -tcc -c 0 -m $MODELNAME.tcc.libsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -tcc -c 0 -m $MODELNAME.tcc.libsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jlibsvm TCC." $BASE/jcp_train.sh -tcc -c 1 -m $MODELNAME.tcc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -tcc -c 1 -m $MODELNAME.tcc.jlibsvmmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training jliblinear TCC." $BASE/jcp_train.sh -tcc -c 2 -m $MODELNAME.tcc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -tcc -c 2 -m $MODELNAME.tcc.jliblinearmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training Neuroph TCC." $BASE/jcp_train.sh -tcc -c 3 -m $MODELNAME.tcc.neurophmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -tcc -c 3 -m $MODELNAME.tcc.neurophmodel $TRAINING_ARGS $TRAINING_SET

echo;
echo "Training DeepLearning4j TCC." $BASE/jcp_train.sh -tcc -c 4 -m $MODELNAME.tcc.deeplearning4jmodel $TRAINING_ARGS $TRAINING_SET
$TIME $BASE/jcp_train.sh -tcc -c 4 -m $MODELNAME.tcc.deeplearning4jmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV SVM TCC." $BASE/jcp_train.sh -tcc -c 3 -m $MODELNAME.tcc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -tcc -c 3 -m $MODELNAME.tcc.ocvsvmmodel $TRAINING_ARGS $TRAINING_SET

#echo;
#echo "Training OpenCV RF TCC." $BASE/jcp_train.sh -tcc -c 4 -m $MODELNAME.tcc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
#$TIME $BASE/jcp_train.sh -tcc -c 4 -m $MODELNAME.tcc.ocvrfmodel $TRAINING_ARGS $TRAINING_SET
