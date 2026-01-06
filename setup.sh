#!/bin/bash

myName="${BASH_SOURCE:-$0}"
myDir=/home/cua/B0_FarForward
myDir=$(readlink -f -- $myDir)

source $myDir/.venv/bin/activate
export PYTHONPATH=$MY_PYTHON_LIB_DIR${PYTHONPATH:+:$PYTHONPATH}