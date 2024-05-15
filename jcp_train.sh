#!/bin/bash
# Expected environment:
#  OCVJARDIR - directory with the OpenCV Java jar archives.
#  OCVLIBDIR - directory with the OpenCV JNI library.


DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

java -Xmx4096m -classpath ${DIR}/build/jar/jcp.jar:${DIR}/../colt/lib/colt.jar:${DIR}/../colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:${DIR}/../JSON/json.jar:${DIR}/../libsvm-java/libsvm.jar:${DIR}/../liblinear-java/liblinear-java.jar:${DIR}/../neuroph-2.98/Framework/*:${DIR}/../neuroph-2.98/Framework/libs/*:${DIR}/../deeplearning4j/* -Djava.library.path=$DIR/lib/:$OCVLIBDIR se.hb.jcp.cli.jcp_train $@
