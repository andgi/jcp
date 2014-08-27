#!/bin/sh
java -Xmx4096m -classpath build/jar/jcp.jar:../colt/lib/colt.jar:../colt/lib/concurrent.jar jcp.cli.jcp_train $@
