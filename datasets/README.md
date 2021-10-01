## Zenodo

First, download data from Zenodo ([doi:10.5281/zenodo.5348291](https://doi.org/10.5281/zenodo.5348291)). This data should be stored in `Zenodo/v1/`.

Now `Zenodo/v1/` should contain the following:
- **ams_all_preds.csv.gz**: The AMS dataset predictions when using an RF or baseline model trained on the training dataset. Includes the predicted score and rank from each model for each compound. We started with 8,434,707 AMS compounds and detected that 247,025 were in the LC or MLPCN training data. These were removed from the AMS list, leaving 8,187,682 compounds to score. The compound matching was done on the SMILES that we canonicalized in rdkit.
- **ams_order_results.csv.gz**: Information about the 1,024 compounds purchased from the AMS library. Excludes the 4 AMS compounds that were incompletely dissolved. Includes the chemical feature representation, information from the vendor, RF and baseline model predictions, screening results, and clustering results.
- **cdd_training_data.tar.gz**: The LC1234 and MLPCN PriA-SSB screening data exported from CDD. These files are described in greater detail in the [`preprocessing`](../preprocessing) directory. 
- **enamine_dose_reponse_curves.tsv**: The dose response curve summaries from all three runs on the 68 Enamine compounds. If a compound was tested multiple times, only the highest-quality dose response curve was used.
- **enamine_final_list.csv.gz**: The final 100 filtered compounds from `enamine_top_10000.csv.gz`. Contains compound information from Enamine as well as RF model scores, chemical feature representations, and clustering results. See [`analysis_notebooks/enamine_final_list/`](../analysis_notebooks/enamine_final_list/).
- **enamine_PriA-SSB_dose_response_data.tar.gz**: The dose response screening data from all three runs on the 68 Enamine compounds.  The 2021-06-16 run was originally screened on 2020-08-24. 2021-06-16 is the date the compound identities were corrected. This run contains two 1,536 well plates.
- **enamine_top_10000.csv.gz**: Top 10,000 predictions from the Enamine REAL dataset using the selected RF model. Contains compound information from Enamine as well as RF model scores, chemical feature representations, and clustering results.
- **master_df.csv.gz**: The output of preprocessing the files in `cdd_training_data.tar.gz`. Contains 441,900 rows. See the [`preprocessing`](../preprocessing) directory for more information.
- **random_forest_classification_139.pkl**: The saved RF classification model with [hyperparameter ID 139](../config/random_forest_classification/139.json). This model was used to score the AMS and Enamine REAL libraries. See the [`src`](../src) directory for code to load the model and make predictions on new compounds.
- **train_ams_real_cluster.csv.gz**: Contains cluster IDs for Taylor-Butina clustering at a 0.4 threshold applied to the training compounds, 1,024 tested AMS compounds, and top-ranked 10,000 compounds from Enamine REAL. Includes the chemical features, dataset to which the compound belongs, leader compounds for each cluster, and whether the compound is a known hit.
- **training_df_single_fold.csv.gz**: This is the ten folds in `training_folds.tar.gz` merged for convenience. Contains 427300 compounds.
- **training_df_single_fold_with_ams_clustering.csv.gz**: Contains cluster ID for Taylor-Butina clustering applied to the Training+AMS_Order(1024) compounds at 0.2, 0.3, and 0.4 thresholds.
- **training_folds.tar.gz**: The LC1234 and MLPCN compound results used for cross validation and model selection. This is the `master_df.csv.gz` that is processed and split into ten folds.

The original chemical screening data are available in [PubChem](https://pubchem.ncbi.nlm.nih.gov/bioassay/1272365).

## patterns

The `patterns/` folder contains files used in pre- and post-processing:

- **Salts.txt**: contains salt patterns to remove from SMILES. This is used in `preprocessing/` by rdkit's [FilterCatalog](https://github.com/rdkit/rdkit/pull/536) and is originally from [rdkit](https://github.com/rdkit/rdkit/blob/master/Data/Salts.txt).
- **ToxAlerts_Schorpp2014_SMARTS.txt**: contains SMARTS patterns for AlphaScreen frequent hitters by [Schorpp 2014](https://journals.sagepub.com/doi/10.1177/1087057113516861). These were specifically gathered from [ToxAlerts](https://pubs.acs.org/doi/10.1021/ci300245q) platform available at [http://ochem.eu/alerts](http://ochem.eu/alerts).
