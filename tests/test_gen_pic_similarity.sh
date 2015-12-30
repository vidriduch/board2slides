#!/bin/bash

RET=0
./project.py $1 -d -o ./test1_

if [ ! -f test1_0.png ]; then
    exit 1
fi

# we have to use dct because radial has some problem with .png images
VAR=$(./toolkit.py ./test1_0.png --test-similarity $2 -a dct)

if [ "${VAR}" == "False" ]; then
    RET=1
fi

# clean up
rm mask.png &> /dev/null
rm test1_0.png &> /dev/null
rm test1_0_meta.csv &> /dev/null

exit $RET
