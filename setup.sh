#!/bin/bash

#unused alternative: ln -s YOURS_PATH_TO_cuDNN/libcudnn.so.7.0.3 YOURS_PATH_TO_cuDNN/libcudnn.so.6

echo 'PATH is:'
echo $PATH
echo 'LD_LIBRARY_PATH is:'
echo $LD_LIBRARY_PATH
echo "lib64:"
ls /usr/local/cuda/lib64/ | grep "libcudn*"
echo "include: "
ls /usr/local/cuda/include/ | grep "libcudn*"

wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE} -P /root/downloads && \
cd /root/downloads && \
tar -xzvf ${CUDNN_TAR_FILE} && \
cp cuda/include/cudnn.h /usr/local/cuda-8.0/include && \
cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/ && \
chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

cd /usr/local/cuda/lib64 && \
rm libcudnn.so && \
rm libcudnn.so.6 && \
ln libcudnn.so.6.* libcudnn.so.6 && \
ln libcudnn.so.6 libcudnn.so && \
ldconfig

echo 'PATH is:'
echo $PATH
echo 'LD_LIBRARY_PATH is:'
echo $LD_LIBRARY_PATH
echo "lib64:"
ls /usr/local/cuda/lib64/ | grep "libcudn*"
echo "include: "
ls /usr/local/cuda/include/ | grep "libcudn*"

which pip
whiich pip3

pip3 install cudnn-python-wrappers
pip3 install tf-nightly-gpu