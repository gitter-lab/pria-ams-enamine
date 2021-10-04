# Virtual screening on PriA-SSB with the AMS and Enamine REAL libraries

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5348291.svg)](https://doi.org/10.5281/zenodo.5348291)

## Citation

If you use this software or the high-throughput screening data, please cite:

Moayad Alnammi, Shengchao Liu, Spencer S. Ericksen, Gene E. Ananiev, Andrew F. Voter, Song Guo, James L. Keck, F. Michael Hoffmann, Scott A. Wildman, Anthony Gitter.
Evaluating scalable supervised learning for synthesize-on-demand chemical libraries.
2021.

**Preprint coming soon**

## Setup

- Install [Anaconda](https://www.anaconda.com/download/).
- Clone or download this repository.
- Create conda environment from `cpu_env.yml`. Note that the CPU environment is only compatible on Linux due to the `xgboost=0.80` package.
```
conda env create -f cpu_env.yml
conda activate pria_ams_enamine_cpu
```

The conda environment in `cpu_env.yml` is not Windows-compatible.
The neural network-based models use the `gpu_env.yml` conda environment instead and use Keras with the Theano backend.

## Repository contents

### analysis_notebooks

Contains Jupyter notebooks that analyze the AMS and Enamine results.
In addition to looking at the number of new hits, it also analyzes the chemical structures between new hits and training set hits.

### chtc

Contains shell scripts for training the models in `src/` on the train-folds and then computing performance on the test-fold at the Center for High-Throughput Computing (CHTC).
The results are used for cross-validation and model selection.

### config

Contains json config files for model hyperparameters.

### datasets

A directory for the training and prospective compound datasets, which can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.5348291).
Also conatins chemical pattern files used in processing.

### output

Contains output results from models during the cross validation and model selection stages. 
These only include the top-20 (with ties) models from the cross validation stage from each model class.
Also contains output predictions on AMS compounds for the prospective stage.

### predict_REAL_db

Contains source code for generating prediction files for the Enamine REAL DB dataset.
The file `predict_real_db.py` processes the Enamine REAL dataset in parts via the `real_db_file` argument.

### preprocessing

Contains source code and description for the preprocessing steps on the Life Chemicals and MLPCN libraries used for model training. 
The resulting dataset is used in the training pipeline described in the paper. 

### preprocessing4aldrich

Contains source code and scripts for preprocessing the AMS library in a similar fashion to the training dataset. 
This is to ensure that the features are generated in the same manner. 

### src

Contains source code for the virtual screening models and for scoring compounds in the AMS library.

## Additional information

### Third-party data
This repository contains third-party data and code.
See the [`datasets`](datasets#patterns) and [`analysis_notebooks/enamine_final_list`](analysis_notebooks/enamine_final_list) directories for details and attribution.

### Taylor-Butina implementation

The Taylor-Butina implementation can be found [here](https://github.com/gitter-lab/active-learning-drug-discovery/blob/master/active_learning_dd/utils/generate_bt_clustering.py).
