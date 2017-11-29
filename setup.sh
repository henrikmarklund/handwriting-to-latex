#!/bin/bash

echo 'PATH:'
echo $PATH

echo 'LD_LIBRARY_PATH'
echo $LD_LIBRARY_PATH

echo "lib64"
ls /usr/local/cuda/lib64/ | grep "libcud*"

echo "include"
ls /usr/local/cuda/include/ | grep "libcud*"

#ln -s YOURS_PATH_TO_cuDNN/libcudnn.so.7.0.3 YOURS_PATH_TO_cuDNN/libcudnn.so.6