This file compiles notes and remarks for merging LC1234 and MLPCN datasets.

## Merging Raw PriA-SSB Screening Datasets
All files involved in the merge are in UW-Madison Box Storage folder named: `LC1234_MLPCN_Clean`. A total of **five** PriA-SSB screening datasets were merged:

1. `CDD CSV Export - LC123 Primary.csv`: **primary** screen for `74,896` LC123 Library molecules.
2. `CDD CSV Export - LC123 Retest.csv`: **secondary** screen for `2,698` LC123 Library molecules.
3. `CDD CSV Export - LC4 Primary.csv`: **primary** screen for `25,290` LC4 Library molecules.
4. `CDD CSV Export - MLPCN Primary.csv`: **primary** screen for `337,990` MLPCN Library molecules.
5. `CDD CSV Export - MLPCN Retest.csv`: **secondary** screen for `1,400` MLPCN Library molecules.

All these files have 8 columns: [`SMSSF ID`, `CDD SMILES`, `Batch Name`, `Library ID`, `Plate Name`, `Plate Well`, `Run Date`, `PriA-SSB AS % inhibition`]. Description for these can be found in the next section.

The merged file is named: `merged_cdd_2018_10_15.csv.gz` containing `442,274` molecules.

# Preprocessing Merged Dataset

## Master Dataframe 
Create a master dataframe with the following columns:

1. `Molecule ID`: uniquely identifies a molecule. Currently a non-negative integer.
2. `Duplicate ID`: denotes the duplicate number of a molecule. This is done so that the same molecule can have multiple % inhibition readings; allows grouping by `Molecule ID`.
3. `SMSSF ID`: molecule ID used by SMSSF for easy cross-referencing and issue followup.
4. `Batch Name`: used by SMSSF for further batch identification of screens. Not used for any purpose in this project. 
5. `Library ID`: the library the molecule is from; i.e. LC1, LC3, MLPCN, etc.
6. `Plate Name`: specifies the plate identifier for this molecule. Can help identify retests if `Plate Name` contains `CP`.
7. `Plate Well`: specifies the well row-column combination of the molecule.
8. `Run Date`: specifies the date the data was entered into CDD.
9. `CDD SMILES`: the CDD SMILES of the molecule.
10. `rdkit SMILES`: rdkit canonical smiles of the molecule.
11. `MorganFP`: rdkit Morgan fingerprint of the molecule.
12. `PriA-SSB AS % inhibition`: the % inhibition score of the molecule.
13. `PriA-SSB AS Activity`: the binary activity {0, 1} of the molecule. See the binarization rules section.
14. `Primary Filter`: is active if the median of the primary screens for this molecule are greater than or equal `binary_threshold`.
15. `Retest Filter`: is active if the median of the retest screens for this molecule are greater than or equal `binary_threshold`.
16. `PAINS Filter`: is active if the molecule passes the rdkit PAINS filter.

### Duplication Rules:
Entries in the master dataframe should not match in the following columns: `['SMSSF ID', 'Plate Name', 'Plate Well', 'Run Date', 'PriA-SSB AS % inhibition']`. If two entries match on these columns, one of them is removed. 

Note that some molecules can have different `SMSSF ID`, but the same `rdkit SMILES`. Therefore, two molecules are duplicates (i.e. should have the same `Molecule ID`, but different `Duplicate ID`) if they have the same `SMSSF ID` OR same `rdkit SMILES`.

## Preprocessing Strategy:
The steps of the preprocessing can be summarized as follows:

0. Read in `merged_cdd_2018_10_8.csv`.
1. Remove molecules with % inhibition <= -100.0.
2. Remove NaNs. Some molecules from CDD had `SMSSF ID` present, but other entries like `CDD SMILES`, `Plate Name`, etc. missing.
3. Define unique identifiers for each row to  `['SMSSF ID', 'Plate Name', 'Plate Well', 'Run Date', 'PriA-SSB AS % inhibition']`. Assert that there are no duplicates on the uniqueness columns.
4. Add `rdkit SMILES` and fingerprints. Note salts are removed using rdkit [SaltRemover](https://www.rdkit.org/docs/source/rdkit.Chem.SaltRemover.html) and [Salts.txt](https://github.com/rdkit/rdkit/blob/master/Data/Salts.txt).
5. Add `Molecule ID` and `Duplicate ID` placeholders. Group molecules that have the same `SMSSF ID` OR `rdkit SMILES` giving them the same `Molecule ID`  and increasing `Duplicate ID`.
6. Generate binary labels `PriA-SSB AS Activity` according to binarization rules section.
7. Finally save the Master DF named: `master_mlpcn_lc_2018_10_18.csv.gz` containing `441,900` molecules.

## Binary Activity Rules
Some molecules can have up to **four** % inhibition scores. How should binary activity labels be generated? 

From discussions, the following rules/steps were defined:

1. The median % inhibition value over all **primary** screens of the molecule is >= 35%
2. The median % inhibition value over all **retest/secondary** screens of the molecule is >= 35%
3. The molecule does not match a PAINS filter. Uses rdkit's [FilterCatalog](https://github.com/rdkit/rdkit/pull/536).

This is done for all molecules individually by grouping using the `Molecule ID`, then applying the above steps. Finally, note that this binary activity label is recorded to ALL entries of the molecule identified by `Molecule ID`. 

# Generating Training Dataframe
The Master DF can have many % inhibition readings for a single molecule. The training dataframe will contain a single entry for each unique molecule identified by its `Molecule ID`. It is generated as follows:

1. Read in the Master DF.
2. Remove retests, leaving the primary screens. Recall that the binary activity is still recorded in the primary entries.
3. Standardize `Library ID`s for easy grouping by libraries followed by stratifying.
4. Group by `Molecule ID` and compute the median for primary screens of each molecule in `PriA-SSB AS % inhibition (Primary Median)` column (and appropriate entries for the other columns; see [notebook](https://github.com/gitter-lab/zinc/blob/master/preprocessing/Stratify%20Sample%20Master%20DF.ipynb)).
5. Save the Training DF. Note this dataframe only has 10 columns: `[Molecule ID, SMSSF ID, Library ID, rdkit SMILES, MorganFP, PriA-SSB AS % inhibition (Primary Median), PriA-SSB AS Activity, Primary Filter, Retest Filter, PAINS Filter]`.

## Stratifying Training DF into 10-folds
One simple method is to ignore the `Library ID` and just stratify sample based on `PriA-SSB AS Activity` into 10-folds.

Another method is to group molecules by the `Library ID` and stratify sample each of these groups into the 10-folds. 

Both of these 10-folds strategies are generated as seen in [notebook](https://github.com/gitter-lab/zinc/blob/master/preprocessing/Stratify%20Sample%20Master%20DF.ipynb).

## Misc. Notes/Remarks
