#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

java -Xmx4096m -classpath ${DIR}/build/jar/jcp.jar:${DIR}/../colt/lib/colt.jar:${DIR}/../colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:${DIR}/../JSON/json.jar:${DIR}/../libsvm-java/libsvm.jar:${DIR}/../liblinear-java/liblinear-java.jar jcp.cli.jcp_train $@
