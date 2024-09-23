#!/bin/bash

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..

# The time program gives better info than the builtin time command.
TIME=`which time`

# TEST_SET is expected to be exported before invoking this script.
#TEST_SET=~/projects/BOEL-KK-2013-2015/src/libsvm-datasets/regression/cpusmall_scale-test.txt

for m in $MODELNAME.icr*; do
  echo
  echo "Testing $m"
  $TIME $BASE/jcp_predict.sh -r -m $m -s 0.05 $TEST_SET
done
