#!/bin/bash
# Expected environment:
#  none

. ./jcp_base.sh

java -classpath ${JCLASSES} se.hb.jcp.cli.jcp_cat $@
