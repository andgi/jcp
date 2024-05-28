#!/bin/bash
# Expected environment:
#  OCVJARDIR - directory with the OpenCV Java jar archives.
#  OCVLIBDIR - directory with the OpenCV JNI library.

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR=$BASE/../test-install

JCLASSES=${BASE}/build/jar/jcp.jar:${DIR}/colt/lib/colt.jar:${DIR}/colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:${DIR}/JSON/json.jar:${DIR}/libsvm-java/libsvm.jar:${DIR}/liblinear-java/liblinear-java.jar:${DIR}/neuroph-java/Framework/:${DIR}/neuroph-java/Framework/neuroph-core-2.98.jar:${DIR}/neuroph-java/Framework/neuroph-contrib-2.98.jar:${DIR}/neuroph-java/Framework/${DIR}/neuroph-java/Framework/neuroph-imgrec-2.98.jar:${DIR}/neuroph-java/Framework/libs/ajt-2.9.jar:${DIR}/neuroph-java/Framework/libs/logback-classic-1.0.13.jar:${DIR}/neuroph-java/Framework/libs/logback-core-1.1.2.jar:${DIR}/neuroph-java/Framework/libs/slf4j-api-1.7.5.jar:${DIR}/neuroph-java/Framework/libs/slf4j-nop-1.7.6.jar:${DIR}/neuroph-java/Framework/libs/visrec-api-1.0.0.jar:${DIR}/deeplearning4j/testdl4j.jar:${DIR}/weka/weka.jar:${DIR}/weka/SMOTE.jar
