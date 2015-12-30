#!/bin/bash

RET=0
./project.py $1 -d -o ./test1_ || exit 1
./toolkit.py ./test1_0.png > /dev/null || exit 1

IFS=',' read -a val < "./test1_0_meta.csv"
if ((${val[0]} > 10 || 408 > ${val[1]} > 428 ||
     ${val[2]} > 12 || 121 > ${val[3]} > 141))
then
    RET=1
fi

# clean up
rm mask.png &> /dev/null
rm test1.0.png &> /dev/null
rm test1_0_meta.csv &> /dev/null

exit $RET
