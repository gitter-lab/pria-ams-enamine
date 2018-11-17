#!/bin/bash

echo Cluster $cluster
echo Process $process
echo RunningOn $runningon

echo "Start environment configuration."
export HOME=$PWD

wget -q –retry-connrefused –waitretry=10 https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
chmod 777 *
./Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda > /dev/null
export PATH=$PWD/anaconda/bin:$PATH
echo 'Done installing anaconda'
chmod 777 *

conda install --yes -c conda-forge scikit-learn  > /dev/null
conda install --yes -c rdkit rdkit-postgresql > /dev/null
echo 'Done installing libraries'
chmod 777 -R ./anaconda

cd zinc
tar -xzvf config.tar.gz
cd src

echo process "$process"


date
pyexit=0
mode=random_forest_classification
for ix in `seq 0 4`;
do
    mkdir -p ../model_weight/cross_validation_keck/"$mode"
    mkdir -p ../output/cross_validation_keck/"$mode"

    python cross_validation_keck.py \
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