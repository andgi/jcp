#!/bin/bash
# Expected environment:
#  OCVJARDIR - directory with the OpenCV Java jar archives.
#  OCVLIBDIR - directory with the OpenCV JNI library.


DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# LD_LIBRARY_PATH - JCP JNI library and OCV JNI library (might not be needed).
export LD_LIBRARY_PATH=$DIR/lib/:$OCVLIBDIR

java -Xmx4096m -classpath ${DIR}/build/jar/jcp.jar:${DIR}/../colt/lib/colt.jar:${DIR}/../colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:${DIR}/../JSON/json.jar:${DIR}/../libsvm-java/libsvm.jar:${DIR}/../liblinear-java/liblinear-java.jar se.hb.jcp.cli.jcp_train $@
