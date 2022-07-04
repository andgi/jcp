#!/bin/bash

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# The time program gives better info than the builtin time command.
TIME=`which time`

# TEST_SET is expected to be exported before invoking this script.
#TEST_SET=~/projects/BOEL-KK-2013-2015/src/pisvm-datasets/classification/mnist_test_576_rbf_8vr.500.dat

for m in $MODELNAME.icc*; do
  echo
  echo "Testing $m"
  $TIME $BASE/jcp_predict.sh -m $m -s 0.05 $TEST_SET
done
