#!/usr/bin/env bash

for i in {0..12}; do
    echo $i
    KERAS_BACKEND=theano \
    python ensemble_02.py \
    --weight_file=../output/ensemble_02/ensemble_$i.pt \
    --config_json_file=../config/ensemble/"$i".json > ../output/ensemble_02/"$i".out
done
