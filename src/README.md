# Stage 1

```
model_list=[single_deep_classification single_deep_regression xgboost_classification xgboost_regression random_forest_classification]
process_num_list=[0, 1, 2, 3, 4]

python cross_validation_keck.py \
--model="$model" \
--process_num=$process_num \
--weight_file=../model_weight/cross_validation_keck/"$mode"/"$mode"_"$process"_"$ix".pkl \
--config_json_file=../config/cross_validation_keck/"$mode"/"$process".json \
> ../output/cross_validation_keck/"$mode"/"$mode"_"$process"_"$ix".out
```

# Stage 2

run `bash run_ensemble.sh` or following

```
for i in {0..12}; do
    echo $i
    KERAS_BACKEND=theano \
    python ensemble.py \
    --weight_file=../output/ensemble/ensemble_$i.pt \
    --config_json_file=../config/ensemble/"$i".json > ../output/ensemble/"$i".out
done
```

# Stage 3

`python final_stage.py > final.out`
