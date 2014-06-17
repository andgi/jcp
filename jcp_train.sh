#!/bin/sh
java -classpath build/jar/jcp.jar:../colt/lib/colt.jar:../colt/lib/concurrent.jar jcp.cli.jcp_train $@
