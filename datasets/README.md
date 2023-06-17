## Zenodo

First, download data from Zenodo ([doi:10.5281/zenodo.5348290](https://doi.org/10.5281/zenodo.5348290)). This data should be stored in `Zenodo/v1/`.

Now `Zenodo/v1/` should contain the following:
- **ams_all_preds.csv.gz**: The AMS dataset predictions when using an RF or baseline model trained on the training dataset. Includes the predicted score and rank from each model for each compound. We started with 8,434,707 AMS compounds and detected that 247,025 were in the LC or MLPCN training data. These were removed from the AMS list, leaving 8,187,682 compounds to score. The compound matching was done on the SMILES that we canonicalized in rdkit.
- **ams_order_results.csv.gz**: Information about the 1,024 compounds purchased from the AMS library. Excludes the 4 AMS compounds that were incompletely dissolved. Includes the chemical feature representation, information from the vendor, RF and baseline model predictions, screening results, and clustering results.
- **baseline_weight.npy**: The saved Similarity Baseline model, which consists of the active compounds in the training data. This model was used to score the AMS library. See the [`src`](../src) directory for code to load the model and make predictions on new compounds.
- **cdd_training_data.tar.gz**: The LC1234 and MLPCN PriA-SSB screening data exported from CDD. These files are described in greater detail in the [`preprocessing`](../preprocessing) directory.
- **enamine_costs_clustered_v3_with_nneighbor.csv.gz**: Contains 5,620 Enamine compounds that were selected based on the RF prediction score and availability. This file also contains the Taylor-Butina cluster ID when clustering the training compounds, 1,024 tested AMS compounds, and top-ranked Enamine compounds at a 0.4 threshold. The nearest neighbor compounds in the training and AMS sets are also included along with compound information from Enamine, RF model scores, and chemical feature representations.
- **enamine_dose_response_curve_plots.xlsx**: Images of the dose response curves from all three runs on the 68 Enamine compounds. If a compound was tested multiple times, multiple curves are shown in the same plot. The compound structure images and SMILES are exported from CDD, not generated with RDKit.
- **enamine_dose_response_curves.tsv**: The dose response curve summaries from all three runs on the 68 Enamine compounds. If a compound was tested multiple times, only the highest-quality dose response curve was used.
- **enamine_final_list.csv.gz**: The final 100 filtered compounds from `enamine_top_10000.csv.gz`. Contains compound information from Enamine as well as RF model scores, chemical feature representations, and clustering results. See [`analysis_notebooks/enamine_final_list/`](../analysis_notebooks/enamine_final_list/).
- **enamine_PriA-SSB_dose_response_data.tar.gz**: The dose response screening data from all three runs on the 68 Enamine compounds.  The 2021-06-16 run was originally screened on 2020-08-24. 2021-06-16 is the date the compound identities were corrected. This run contains two 1,536 well plates.
- **enamine_top_10000.csv.gz**: Top 10,000 predictions from the Enamine REAL dataset using the selected RF model. Contains compound information from Enamine as well as RF model scores, chemical feature representations, and clustering results.
- **master_df.csv.gz**: The output of preprocessing the files in `cdd_training_data.tar.gz`. Contains 441,900 rows. See the [`preprocessing`](../preprocessing) directory for more information.
- **random_forest_classification_139.pkl**: The saved RF classification model with [hyperparameter ID 139](../config/random_forest_classification/139.json). This model was used to score the AMS and Enamine REAL libraries. See the [`src`](../src) directory for code to load the model and make predictions on new compounds.
- **train_ams_real_cluster.csv.gz**: Contains cluster IDs for Taylor-Butina clustering at a 0.4 threshold applied to the training compounds, 1,024 tested AMS compounds, and top-ranked compounds from Enamine. Includes the chemical features, dataset to which the compound belongs, leader compound for each cluster, and whether the compound is a known hit.
- **training_df_single_fold.csv.gz**: This is all ten folds in `training_folds.tar.gz` merged for convenience. Contains 427,300 compounds.
- **training_df_single_fold_with_ams_clustering.csv.gz**: Contains cluster IDs for Taylor-Butina clustering applied to the 427,300 training compounds and the 1,024 tested AMS compounds. Different clustering results are shown at the 0.2, 0.3, and 0.4 thresholds. Includes the leader compound for each cluster. Although the training and AMS compounds were clustered jointly, only the training compounds' clusters are shown.  The AMS compounds' clusters are in `ams_order_results.csv.gz`.
- **training_folds.tar.gz**: The LC1234 and MLPCN training data split into ten folds. This dataset with 427,300 compounds was used for cross validation and model selection. This dataset is derived from `master_df.csv.gz` as described in the [`preprocessing`](../preprocessing) directory.

The original chemical screening data are available in [PubChem](https://pubchem.ncbi.nlm.nih.gov/bioassay/1272365).

## patterns

The `patterns/` folder contains files used in pre- and post-processing:

- **Salts.txt**: contains salt patterns to remove from SMILES. This is used in `preprocessing/` by rdkit's [FilterCatalog](https://github.com/rdkit/rdkit/pull/536) and is originally from [rdkit](https://github.com/rdkit/rdkit/blob/master/Data/Salts.txt).
- **ToxAlerts_Schorpp2014_SMARTS.txt**: contains SMARTS patterns for AlphaScreen frequent hitters by [Schorpp 2014](https://journals.sagepub.com/doi/10.1177/1087057113516861). These were specifically gathered from [ToxAlerts](https://pubs.acs.org/doi/10.1021/ci300245q) platform available at [http://ochem.eu/alerts](http://ochem.eu/alerts).
