## Zenodo

First, download data from Zenodo ([doi:10.5281/zenodo.5348291](https://doi.org/10.5281/zenodo.5348291)). This data should be stored in `Zenodo/v1/`.

Now `Zenodo/v1/` should contain the following:  
- **cdd_training_data.tar.gz**: The LC1234 and MLPCN raw result files against PriA-SSB target. These are used in preprocessing. 
- **master_df.csv.gz**: The output of preprocessing the files in `cdd_training_data.tar.gz`. Contains 441900 rows. See `preprocessing/` for more info. 
- **training_folds.tar.gz**: The LC1234 and MLPCN compound results used for cross validation and model selection. This is the `master_df.csv.gz` that is processed and split into ten folds.
- **training_df_single_fold.csv.gz**: This is the ten folds in `training_folds.tar.gz` merged for convenience. Contains 427300 compounds.

- **ams_order_results.csv.gz**: Results for the 1024 purchased AMS library compound against the PriA-SSB target.

- **enamine_top_10000.csv.gz**: Top-10000 predictions from the Enamine REAL dataset using the final selected RF model after training on the training dataset. 
- **enamine_final_list.csv.gz**: Contains the final 100 filtered compounds from `enamine_top_10000.csv.gz`. See `analysis_notebooks/enamine_final_list/`.

- **training_df_single_fold_with_ams_clustering.csv.gz**: Contains cluster ID for Taylor-Butina clustering applied to the Training+AMS_Order(1024) compounds at 0.2, 0.3, and 0.4 thresholds.
- **train_ams_real_cluster.csv.gz**: ontains cluster ID for Taylor-Butina clustering applied to the Training+AMS_Order(1024)+top10kenamine(10k) compounds at 0.4 threshold.

- **enamine_database.csv.gz**: the Enamine REAL dataset used in this paper.
- **ams_all_preds.csv.gz**: The AMS dataset predictions when using an RF or baseline model trained on the training dataset. We started with 8434707 AMS compounds and detected that 247025 were in the LC or MLPCN compound list.  These were removed from the AMS list, leaving 8187682 compounds. The compound matching was done on the SMILES that we canonicalized in rdkit.
- **enamine_PriA-SSB_dose_response_data.tar.gz**: The dose response screening data from all three runs on the 68 Enamine compounds.  The 2021-06-16 run was originally screened on 2020-08-24. 2021-06-16 is the date the compound identities were corrected. This run contains two 1536 well plates.
- **enamine_dose_reponse_curves.tsv**: The dose response curve summaries from all three runs on the 68 Enamine compounds. Only the highest-quality dose response curve per compound was used.

## patterns

The `patterns/` folder contains files used in pre- and post-processing:

- **Salts.txt**: contains salt patterns to remove from SMILES. This is used in `preprocessing/` by rdkit's [FilterCatalog](https://github.com/rdkit/rdkit/pull/536) and is originally from [rdkit](https://github.com/rdkit/rdkit/blob/master/Data/Salts.txt).
- **ToxAlerts_Schorpp2014_SMARTS.txt**: contains SMARTS patterns for AlphaScreen frequent hitters by [Schorpp 2014](https://journals.sagepub.com/doi/10.1177/1087057113516861). These were specifically gathered from [ToxAlerts](https://pubs.acs.org/doi/10.1021/ci300245q) platform available at [http://ochem.eu/alerts](http://ochem.eu/alerts).
