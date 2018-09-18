This file compiles notes and remarks for merging LC1234 and MLPCN datasets.

## Raw Data Files
All files involved in the merge are in UW-Madison Box Storage.

### LC123 data files involved:
1. `lifechem123_cleaned_2017_03_10.smi` - contains SMSSF ID and SMILES info.
2. `screening_smsf_continuous_2017_03_10.xlsx` - contains SMSSF ID and % inhibition.

### LC4 data files involved:
3. `lifechem4_cleaned_2017_03_10.smi` - contains SMSSF ID and SMILES info.
4. `Keck_LC4_export.xlsx` - contains SMSSF ID and % inhibition.
5. `pria_lc4_retest_may18.csv` - contains SMSSF ID, primary % inhibition, and retest % inhibition.

### MLPCN data files involved:
6. `keck_PriA_MLPCN.csv` - contains SMSSF ID, SMILES, and % inhibition.
7. `keck_retest_continuous.xlsx` - contains SMSSF ID, primary % inhibition, and retest % inhibition.

## Master Dataframe 
Create a master dataframe with the following columns:
1. `Molecule ID`: uniquely identifies a molecule. Currently an non-negative integer.
2. `Duplicate ID`: denotes the duplicate number of a molecule. This is done so that the same molecule can have multiple % inhibition readings; allows grouping by `Molecule ID`.
3. `SMSSF ID`: molecule ID used by SMSSF for easy cross-referencing and issue followup.
4. `Supplier ID`: the supplier of the molecule.
5. `SMILES`: rdkit canonical smiles of the molecule.
6. `MorganFP`: rdkit Morgan fingerprint of the molecule.
7. `PriA-SSB AS normalized % inhibition`: the % inhibition score of the molecule.
8. `PriA-SSB AS Activity`: the binary activity {0, 1} of the molecule. 

### Duplication Rules:
Two molecules are duplicates (i.e. should have the same `Molecule ID`, but different `Duplicate ID`) if they have different `PriA-SSB AS normalized % inhibition` AND any of the following holds:
1. They have the same `SMSSF ID`.
2. They have the same `SMILES` i.e. the same rdkit canonical smiles.

If two molecules have the same 'SMSSF ID', 'SMILES', AND 'PriA-SSB AS normalized % inhibition', then they are regarded as the same entry; one of them is removed.

## Merging Strategy:
The steps of the merging can be summarized as follows:
1. Process LC123 files: get SMSSF ID and SMILES, then record the % inhibition. Ensure duplication rules are applied to the Master DF.
2. Process LC4 files: cross-reference LC4 molecules with the current Master DF; adding new molecules and duplicates. Add retests with different `Duplicate ID`.
3. Process MLPCN files: cross-reference MLPCN molecules with the current Master DF; adding new molecules and duplicates. Add retests with different `Duplicate ID`.
4. Finalize Master DF by removing duplicate ROWS based on 'SMSSF ID', 'SMILES', AND 'PriA-SSB AS normalized % inhibition'.
5. Two molecules have missing SMILES (SMSSF-0046450 and SMSSF-0060022). Add their SMILES and FPS as discussed:

|SMSSF ID       |  SMILES                 |
|---------------|----------------------------------------------------------------|
| SMSSF-0046450 | `COc1ccc(cc1NC(=O)CS(=O)c2ncc(n2C)c3ccccc3)C`                   |
| SMSSF-0060022 | `C1CCCN(CC1)c2c(nc(o2)c3ccccc3)[P+](c4ccccc4)(c5ccccc5)c6ccccc6` |

## Binary Activity Rules
Some molecules can have up to 4 % inhibition scores. How should binary activity labels be generated? 

From discussions, the following rules were advised:
1. Group molecules by `Molecule ID` and apply aggregate function: `median`.
2. Apply threshold on the `median` to generate binary label.

Still need to add PAINS filter results. Use rdkit's [FilterCatalog](https://github.com/rdkit/rdkit/pull/536) for such filters.