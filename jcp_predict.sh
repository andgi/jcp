#!/bin/sh
java -Xmx8192m -classpath build/jar/jcp.jar:../colt/lib/colt.jar:../colt/lib/concurrent.jar:$OCVJARDIR/opencv.jar:../JSON/json.jar:../libsvm-java/libsvm.jar:../liblinear-java/liblinear-java.jar jcp.cli.jcp_predict $@
