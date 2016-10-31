#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

$DIR/test-icc.sh
echo
echo
$DIR/test-lcicc.sh
echo
echo
$DIR/test-tcc.sh
echo
echo
$DIR/test-lctcc.sh
