#!/bin/bash
# Expected environment:
#  OCVJARDIR - directory with the OpenCV Java jar archives.
#  OCVLIBDIR - directory with the OpenCV JNI library.

. ./jcp_base.sh

java -Xmx4096m -classpath $JCLASSES -Djava.library.path=$BASE/lib/:$OCVLIBDIR se.hb.jcp.cli.jcp_train $@
