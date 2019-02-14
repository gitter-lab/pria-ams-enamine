#!/usr/bin/env bash
model_list=(single_deep_classification single_deep_regression  random_forest_classification random_forest_regression xgboost_classification xgboost_regression)

for model in "${model_list[@]}"; do
    python cross_validation_keck.py --weight_file=temp.pt --process_num=1 --model="$model" --config_json_file=../config/cross_validation_keck/"$model"/2.json > "$model".out
done
