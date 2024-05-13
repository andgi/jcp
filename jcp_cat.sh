#!/bin/bash
# Expected environment:
#  None

BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIR=$BASE/../test-install

java -classpath ${BASE}/build/jar/jcp.jar:${DIR}/colt/lib/colt.jar:${DIR}/colt/lib/concurrent.jar:${DIR}/JSON/json.jar se.hb.jcp.cli.jcp_cat $@
