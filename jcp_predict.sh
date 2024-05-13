#!/bin/bash
# Expected environment:
#  OCVJARDIR - directory with the OpenCV Java jar archives.
#  OCVLIBDIR - directory with the OpenCV JNI library.

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR=$BASE/../test-install

java -Xmx8192m -classpath ${BASE}/build/jar/jcp.jar:${DIR}/colt/lib/colt.jar:${DIR}/colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:${DIR}/JSON/json.jar:${DIR}/libsvm-java/libsvm.jar:${DIR}/liblinear-java/liblinear-java.jar -Djava.library.path=$BASE/lib/:$OCVLIBDIR se.hb.jcp.cli.jcp_predict $@
