#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

$DIR/train-icc.sh
echo
echo
$DIR/train-lcicc.sh
echo
echo
$DIR/train-tcc.sh
echo
echo
$DIR/train-lctcc.sh


