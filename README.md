# Virtual screening on PriA-SSB with the AMS and Enamine REAL libraries

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5348291.svg)](https://doi.org/10.5281/zenodo.5348291)

## Citation

If you use this software or the high-throughput screening data, please cite:

**Preprint coming soon**

## Installation

- Install [Anaconda](https://www.anaconda.com/download/).
- Clone or download this repository.
- Create conda environment from `cpu_env.yml`. Note that the CPU environment is only compatible on Linux due to the `xgboost=0.80` package.
```
conda env create -f cpu_env.yml
source activate pria_ams_enamine_cpu

Note that the neural network based models use the `gpu_env.yml` conda environment and makes use of Keras with the Theano backend. 

## datasets

Contains training and prospective datasets, and pattern files used in processing. See the README in the folder for more info.

## src

Contains source code for models and stage runs.

## output

Contains output results from models during the cross validation and model selection stage. 
These only include the top-20 (with ties) models from the cross validation stage from each model class.
Also contains output predictions on AMS compounds for the prospective stage.

## analysis_notebooks

Contains Jupyter notebooks that analyzes AMS and Enamine results. In addition to looking at number of new hits, it also analyzes the structures beteween new hits and training set hits.


## chtc

Contains shell scripts for training the models in `src/` on the train-folds and then computing performance on the test-fold. The results are used for cross-validation and model selection.

## config

Contains json config files for model hyperparameters.

## predict_REAL_db

Contains source code for generating prediction files for the Enamine REAL DB dataset. 
The file `predict_real_db.py` processes the Enamine REAL dataset in parts via the `real_db_file` argument. 

## preprocessing

Contains source code and description for the preprocessing steps on the LC1234 and MLPCN libraries. 
The resulting 'Master DF' dataset is used in the training pipeline described in the paper. 


## preprocessing4aldrich

Contains source code for preprocessing the raw AMS library in similar fashion to the training dataset. 
This is to ensure that the features are generated in the same manner. 

## Taylor-Butina Implementation

The Taylor-Butina implementation can be found [here](https://github.com/gitter-lab/active-learning-drug-discovery/blob/master/active_learning_dd/utils/generate_bt_clustering.py).
