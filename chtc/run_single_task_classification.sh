#!/bin/bash

echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

mkdir -p "$transfer_output_files"/cross_validation_keck


export HOME=$PWD

export PATH=$PATH:/usr/local/cuda-8.0/bin

wget -q --retry-connrefused --waitretry=10 https://repo.continuum.io/miniconda/Miniconda2-4.2.12-Linux-x86_64.sh
bash Miniconda2-4.2.12-Linux-x86_64.sh -b -p ./anaconda
export PATH=$PWD/anaconda/bin:$PATH
chmod 777 *
chmod 777 -R ./anaconda

cd zinc
conda env create -n zinc_project -f gpu_env.yml
source activate zinc_project
echo 'Done loading environment.'
tar -xzvf config.tar.gz
cd ..


wget -nv http://proxy.chtc.wisc.edu/SQUID/agitter/cudnn/cudnn-8.0-linux-x64-v5.0-ga.tgz
tar -xf cudnn-8.0-linux-x64-v5.0-ga.tgz
export CUDNN_PATH=$(pwd)/cuda
export LD_LIBRARY_PATH=$(pwd)/cuda/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# configure keras
mkdir -p .keras
echo '{"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' > .keras/keras.json

# configure theano
echo -e "[global]\ndevice = gpu\nfloatX = float32\nbase_compiledir = ./tmp\n\n[gpuarray]\npreallocate = 0.8\n\n[dnn]\nlibrary_path=$CUDNN_PATH/lib64\ninclude_path=$CUDNN_PATH/include\nenabled=True" > .theanorc

nvidia-smi

# Test Keras import
KERAS_BACKEND=theano python -c "from keras import backend"
pyexit=$?
echo "$pyexit"

cd zinc/src

mode=single_deep_classification

for ix in `seq 0 4`;
do
    mkdir -p ../model_weight/cross_validation_keck/"$mode"
    mkdir -p ../output/cross_validation_keck/"$mode"

    KERAS_BACKEND=theano python cross_validation_keck.py \
    --config_json_file=../config/cross_validation_keck/"$mode"/"$process".json \
    --weight_file=../model_weight/cross_validation_keck/"$mode"/"$mode"_"$process"_"$ix".pkl \
    --process_num="$ix" \
    --model="$mode" > ../output/cross_validation_keck/"$mode"/"$mode"_"$process"_"$ix".out

    pyexit=$?
    echo "$pyexit"
done
date

cp -r ../output/cross_validation_keck ~/"$transfer_output_files"/
cp -r ../model_weight/cross_validation_keck ~/"$transfer_output_files"/

exit $pyexit
