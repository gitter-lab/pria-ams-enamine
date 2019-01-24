#!/usr/bin/env bash

for i in {0..12}; do
    echo $i
    KERAS_BACKEND=theano \
    python ensemble.py \
    --weight_file=../output/ensemble/ensemble_$i.pt \
    --config_json_file=../config/ensemble/"$i".json > ../output/ensemble/"$i".out
done
