# Model Files

Models used in paper: baseline_similarity.py, deep_classification.py (NN-C), deep_regression.py (NN-R), random_forest_classification.py (RF-C), xgboost_classification.py (XGB-C), xgboost_regression.py (XGB-R), ensemble.py (Ensemble Model-based), ensemble_02.py (Ensemble Max-Vote).

Models NOT used in paper: character_rnn_classification.py, grammar_cnn_classification.py, random_forest_regression.py, tree_net.py.

# Helper Files

- evaluation.py: contains metrics code for AUC[ROC], AVG[PR], and NEF. 
- function.py: various data helper functions.
- util.py: output helper functions.
- CallBacks.py: helper callback functions gradient-based models.

# Cross Validation Stage

To run a model in the cross validation stage, see example below:

```
model=random_forest_classification
cv_idx=0
hyperparam_idx=0

python cross_validation_keck.py \
--model="$model" \
--process_num=${cv_idx} \
--weight_file=../output/"$model"/"$model"_"${hyperparam_idx}"_"${cv_idx}".pkl \
--config_json_file=../config/"$model"/"${hyperparam_idx}".json \
> ../output/"$model"/"$model"_"${hyperparam_idx}"_"${cv_idx}".out
```

1. Replace `model` with the desired model class from the following list: 
```
model_list=[single_deep_classification, single_deep_regression, xgboost_classification, xgboost_regression, random_forest_classification]
```
2. Replace `cv_idx` with the desired cross validation ID (0, 1, 2, or 3 as shown in the paper). 
3. Replace `hyperparam_idx` with the desired model class' hyperparameter ID (see config folder). 

The weight and output files will generated in the designated directories.

# Model Selection Stage

For models in the following list:
```
model_list=[single_deep_classification, single_deep_regression, xgboost_classification, xgboost_regression, random_forest_classification]
```

Similar to the cross validation stage script, but the `cv_idx` is restricted to 4:

```
model=random_forest_classification
hyperparam_idx=0

python cross_validation_keck.py \
--model="$model" \
--process_num=4 \
--weight_file=../output/"$model"/"$model"_"${hyperparam_idx}"_4.pkl \
--config_json_file=../config/"$model"/"${hyperparam_idx}".json \
> ../output/"$model"/"$model"_"${hyperparam_idx}"_4.out
```

For Ensemble models, use the following:

```
hyperparam_idx=0
    
KERAS_BACKEND=theano \
python ensemble.py \
--weight_file=../output/ensemble/ensemble_${hyperparam_idx}.pt \
--config_json_file=../config/ensemble/"${hyperparam_idx}".json > ../output/ensemble/"${hyperparam_idx}".out
```

Replace `hyperparam_idx` with the desired ensemble ID (ranges from 0 to 13, see config folder). Replace `ensemble.py` with `ensemble_02.py` for Max-Vote.

# AMS Prospective Stage

To generate predictions for the AMS library using the top-1 RF-C model, first create the `../model_weight/final_stage/` directory:
```
mkdir -p ../model_weight/final_stage/
```

Then run the following to train the RF-C and baseline models, and save the trained models:
```
python prospective_stage.py \
--config_json_file=../config/random_forest_classification/139.json \
--weight_file=../model_weight/final_stage/random_forest_classification_139.pkl \
--model=random_forest_classification \
--mode=training
```
```
python prospective_stage.py \
--config_json_file=../config/baseline_similarity.json \
--weight_file=../model_weight/final_stage/baseline_weight.npy \
--model=baseline \
--mode=training
```

Finally, to generate predictions for the AMS library, run: 

```
python prospective_stage.py \
--config_json_file=../config/random_forest_classification/139.json \
--weight_file=../model_weight/final_stage/random_forest_classification_139.pkl \
--model=random_forest_classification \
--mode=prediction
```
```
python prospective_stage.py \
--config_json_file=../config/baseline_similarity.json \
--weight_file=../model_weight/final_stage/baseline_weight.npy \
--model=baseline \
--mode=prediction
```

The predictions can be found at `../output/final_stage/`.

# Enamine REAL Prospective Stage

The directory [`predict_REAL_db`](../predict_REAL_db) contains code to score compounds in the Enamine REAL database.
This code documents the scoring process but cannot be run because this repository does not contain Enamine's chemical library.
