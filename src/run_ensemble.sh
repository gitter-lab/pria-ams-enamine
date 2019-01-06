#!/usr/bin/env bash

for i in {0..7}; do
    echo $i
    KERAS_BACKEND=theano \
    python ensemble.py \
    --weight_file=temp.pt \
    --config_json_file=../config/ensemble/"$i".json > ../output/ensemble/"$i".out
done
