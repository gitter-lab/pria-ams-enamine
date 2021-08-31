# Virtual screening on PriA-SSB with the AMS and Enamine REAL libraries

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5348291.svg)](https://doi.org/10.5281/zenodo.5348291)

## Citation

If you use this software or the high-throughput screening data, please cite:

**Preprint coming soon**

## Installation

- Install [Anaconda](https://www.anaconda.com/download/).
- Clone or download this repository.
- Create conda environment from `cpu_env.yml`:
```
conda env create -f conda_env.yml
source activate zinc_project
```
- Install repository package from repo's root directory `./setup.py`:
```
pip install -e .
```

Note that the neural network based models use the `gpu_env.yml` conda environment and makes use of Keras with the Theano backend. 

## datasets

The following datasets are available on Zenodo ([doi:10.5281/zenodo.5348291](https://doi.org/10.5281/zenodo.5348291)):
- **raw.zip**: The LC1234 and MLPCN raw files used for preprocessing.
- **master_df.csv.gz**: The results of preprocessing and binarizing the raw files. This is then split into folds. 
- **training_folds.tar.gz**: The LC1234 and MLPCN compound results against PriA-SSB target used for cross validation and model selection split into ten folds.

- **ams_order_results.csv.gz**: Results for the 1024 purchased AMS library compound against the PriA-SSB target.

- **enamine_top_10000.csv.gz**: top-10k predictions from the REAL dataset using an RF model trained on the training dataset. 
- **enamine_final_list.csv.gz**: description later.

- **train_ams_order_cluster.csv.gz**: cluster results when running TB on training set + AMS orders.
- **train_ams_real_cluster.csv.gz**: cluster results when running TB on training set + AMS orders + top10k real.

- **enamine_database.csv.gz**: the Enamine REAL dataset used in this paper.
- **ams_all_preds.csv.gz**: The AMS dataset predictions when using an RF or baseline model trained on the training dataset. We started with 8434707 AMS compounds and detected that 247025 were in the LC or MLPCN compound list.  These were removed from the AMS list, leaving 8187682 compounds. The compound matching was done on the SMILES that we canonicalized in rdkit.
- **enamine_PriA-SSB_dose_response_data.tar.gz**: The dose response screening data from all three runs on the 68 Enamine compounds.  The 2021-06-16 run was originally screened on 2020-08-24. 2021-06-16 is the date the compound identities were corrected. This run contains two 1536 well plates.
- **enamine_dose_reponse_curves.tsv**: The dose response curve summaries from all three runs on the 68 Enamine compounds. Only the highest-quality dose response curve per compound was used.

## src

Contains source code for models and stage runs.

## output

Contains output results from models during the cross validation and model selection stage. 
These only include the top-20 (with ties) models from the cross validation stage from each model class.

## config

Contains json config files for model hyperparameters.

## predict_REAL_db

Contains source code for generating prediction files for the Enamine REAL DB dataset.

## preprocessing

Contains source code and description for the preprocessing steps on the LC1234 and MLPCN libraries. 
The resulting 'Master DF' dataset is used in the training pipeline described in the paper. 

## preprocessing4aldrich

Contains source code for preprocessing the raw AMS library in similar fashion to the training dataset. 
This is to ensure that the features are generated in the same manner. 

## Taylor-Butine Custom Implementation

The Taylor-Butina custom implementation can be found [here](https://github.com/gitter-lab/active-learning-drug-discovery/blob/master/active_learning_dd/utils/generate_bt_clustering.py).
