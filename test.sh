#!/usr/bin/env bash

use_large=$1
if (("$use_large"==1))
then
echo $use_large
else
echo "no thing"
python test2.py
fi
