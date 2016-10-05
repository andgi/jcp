#!/bin/bash
# Expected environment:
#  None

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

java -classpath ${DIR}/build/jar/jcp.jar:${DIR}/../colt/lib/colt.jar:${DIR}/../colt/lib/concurrent.jar:${DIR}/../JSON/json.jar se.hb.jcp.cli.jcp_cat $@
