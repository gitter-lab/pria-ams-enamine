This file compiles notes and remarks for merging LC1234 and MLPCN datasets.

## Raw Data Files
All files involved in the merge are in UW-Madison Box Storage. Now only one file is involved in the merge `merged_cdd_2018_10_8.csv` which includes primary and retest screens from LC1234 and MLPCN.

## Master Dataframe 
Create a master dataframe with the following columns:
1. `Molecule ID`: uniquely identifies a molecule. Currently a non-negative integer.
2. `Duplicate ID`: denotes the duplicate number of a molecule. This is done so that the same molecule can have multiple % inhibition readings; allows grouping by `Molecule ID`.
3. `SMSSF ID`: molecule ID used by SMSSF for easy cross-referencing and issue followup.
4. `Library ID`: the library the molecule is from; i.e. LC1, MLPCN, etc.
5. `Plate Name`: specifies the plate identifier for this molecule. Can help identify retests if `Plate Name` contains `CP`.
6. `Plate Well`: specifies the well row-column combo of the molecule.
7. `Run Date`: specifies the date the data was entered into CDD.
8. `CDD SMILES`: the CDD SMILES of the molecule.
9. `rdkit SMILES`: rdkit canonical smiles of the molecule.
10. `MorganFP`: rdkit Morgan fingerprint of the molecule.
11. `PriA-SSB AS % inhibition`: the % inhibition score of the molecule.
12. `PriA-SSB AS Activity`: the binary activity {0, 1} of the molecule. This is based on the Primary, Retest, and PAINS filters all being active.
13. `Primary Filter`: is active if the median of the primary screens for this molecule are greater than or equal `binary_threshold`.
14. `Retest Filter`: is active if the median of the retest screens for this molecule are greater than or equal `binary_threshold`.
15. `PAINS Filter`: is active if the molecule passes the rdkit PAINS filter.

### Duplication Rules:
Two molecules are duplicates (i.e. should have the same `Molecule ID`, but different `Duplicate ID`) if they have different `PriA-SSB AS % inhibition` AND any of the following holds:
1. They have the same `SMSSF ID`.
2. They have the same `rdkit SMILES` i.e. the same rdkit canonical smiles.

If two molecules have the same `SMSSF ID`, `rdkit SMILES`, AND `PriA-SSB AS % inhibition`, then they are regarded as the same entry; one of them is removed.

## Preprocessing Strategy:
The steps of the preprocessing can be summarized as follows:
0. Read in `merged_cdd_2018_10_8.csv`.
1. Remove molecules with % inhibition <= -100.0.
2. Remove NaNs. Some molecules from CDD had `SMSSF ID` present, but other entries like `CDD SMILES`, `Plate Name`, etc. missing.
3. Define unique identifiers for each row to  `['SMSSF ID', 'Plate Name', 'Plate Well', 'Run Date', 'PriA-SSB AS % inhibition']`. Assert that there are no duplicates on the uniqueness columns.
4. Add `rdkit SMILES` and fingerprints. Note salts are removed using rdkit SaltRemover and Salts.txt.
5. Add `Molecule ID` and `Duplicate ID` placeholders. Group molecules that have the same `SMSSF ID` under same molecule id and increasing duplicate id.
6. Generate binary label according to binarization rules (see below).
7. Finally save the Master DF.

## Binary Activity Rules
Some molecules can have up to 4 % inhibition scores. How should binary activity labels be generated? 

From discussions, the following rules were advised:
1. Group molecules by `Molecule ID` and apply custom aggregate function.
2. The median % inhibition value over all primary screens is >= 35%
3. The compound does not match a PAINS filter
4. The compound has % inhibition >= 35% of the median of retest inhibition

PAINS filter uses rdkit's [FilterCatalog](https://github.com/rdkit/rdkit/pull/536).


## Generating Training Dataframe
The Master DF can have many readings for a single molecule. The training dataframe will contain a single entry for each unique molecule identified by its `Molecule ID`. It is generated as follows:
1. Read in the Master DF.
2. Remove retests.
3. Standardize library ids for easy grouping by libraries followed by stratifying.
4. Group by `Molecule ID` and apply a custom aggregate function that will produce the median for primary screens of each molecule (and appropriate entries for the other columns; see code).
5. Save the Training DF

## Stratifying Training DF into 10-folds
One simple method is to ignore the `Library ID` and just stratify sample based on `PriA-SSB AS Activity` into 10-folds.

Another method is to group molecules by the `Library ID` and stratify sample each of these groups into the 10-folds. 

Both of these 10-folds strategies are generated.

## Misc. Notes/Remarks
