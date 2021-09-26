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
- **cdd_training_data.tar.gz**: The LC1234 and MLPCN raw result files against PriA-SSB target. These are used in preprocessing. 
- **master_df.csv.gz**: The output of preprocessing the files in `cdd_training_data.tar.gz`. Contains 441900 rows. See `preprocessing\` for more info. 
- **training_folds.tar.gz**: The LC1234 and MLPCN compound results used for cross validation and model selection. This is the `master_df.csv.gz` that is processed and split into ten folds.
- **training_df_single_fold.csv.gz**: This is the ten folds in `training_folds.tar.gz` merged for convenience. Contains 427300 compounds.

- **ams_order_results.csv.gz**: Results for the 1024 purchased AMS library compound against the PriA-SSB target.

- **enamine_top_10000.csv.gz**: Top-10000 predictions from the Enamine REAL dataset using the final selected RF model after training on the training dataset. 
- **enamine_final_list.csv.gz**: Contains the final 100 filtered compounds from `enamine_top_10000.csv.gz`. See `analysis_notebooks\enamine_final_list\`.

- **training_df_single_fold_with_ams_clustering.csv.gz**: Contains cluster ID for Taylor-Butina clustering applied to the Training+AMS_Order(1024) compounds at 0.2, 0.3, and 0.4 thresholds.
- **train_ams_real_cluster.csv.gz**: ontains cluster ID for Taylor-Butina clustering applied to the Training+AMS_Order(1024)+top10kenamine(10k) compounds at 0.4 threshold.

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

## Taylor-Butina Implementation

The Taylor-Butina implementation can be found [here](https://github.com/gitter-lab/active-learning-drug-discovery/blob/master/active_learning_dd/utils/generate_bt_clustering.py).
