#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

java -Xmx8192m -classpath ${DIR}/build/jar/jcp.jar:${DIR}/../colt/lib/colt.jar:${DIR}/../colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:${DIR}/../JSON/json.jar:${DIR}/../libsvm-java/libsvm.jar:${DIR}/../liblinear-java/liblinear-java.jar se.hb.jcp.cli.jcp_predict $@
