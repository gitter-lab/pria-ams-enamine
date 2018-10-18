"""
    Merges MLCPN+LC raw datasets into one large master dataframe.
    Refer to "Merging MLCPN and LC datasets - description.txt" for description.
    
    Specify step as 0, 1, 2, or 3. Merging was broken down into 4 steps/jobs since 
    processing all datasets took more than 72 hours (max time for CHTC job). 
    Usage:
        python merge_datasets_chtc.py \
        --rawdata_dir=../datasets/raw/ \
        --output_dir=../datasets/master/ \
        [--FP_size=1024] \
        [--FP_radius=2] \
        [--binary_threshold=35.0]
"""

import argparse
import pandas as pd
import numpy as np
import os
import gzip
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.FilterCatalog import *

"""
    The rules for binarization:
    1) The median % inhibition value over all primary screens is >= 35%
    2) The compound does not match a PAINS filter
    3) The compound has % inhibition >= 35% of the median of retest inhibition
"""
def binarize_rules(grouped_df, pains_catalog, binary_threshold=35.0):
    retest_rows = grouped_df['Plate Name'].str.contains('CP')
    primary_df = grouped_df[~retest_rows] 
    retest_df = grouped_df[retest_rows]
    smiles = grouped_df['rdkit SMILES'].iloc[0]
    
    primary_filter = primary_df['PriA-SSB AS % inhibition'].median() >= binary_threshold
    retest_filter = retest_df['PriA-SSB AS % inhibition'].median() >= binary_threshold
    pains_filter = not pains_catalog.HasMatch(Chem.MolFromSmiles(smiles))
    
    return primary_filter, retest_filter, pains_filter
    
if __name__ == '__main__':
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawdata_dir', action="store", dest="rawdata_dir", required=True)
    parser.add_argument('--output_dir', action="store", dest="output_dir", required=True)
    parser.add_argument('--FP_size', type=int, default=1024, action="store", dest="FP_size", required=False)
    parser.add_argument('--FP_radius', type=int, default=2, action="store", dest="FP_radius", required=False)
    parser.add_argument('--binary_threshold', type=float, default=35.0, action="store", dest="binary_threshold", required=False)
    parser.add_argument('--lb_inh_threshold', type=float, default=-100.0, action="store", dest="lb_inh_threshold", required=False)
    
    given_args = parser.parse_args()
    rawdata_dir = given_args.rawdata_dir
    output_dir = given_args.output_dir
    FP_size = given_args.FP_size
    FP_radius = given_args.FP_radius
    binary_threshold = given_args.binary_threshold
    lb_inh_threshold = given_args.lb_inh_threshold
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    concat_df = pd.read_csv(rawdata_dir+'/merged_cdd_2018_10_15.csv.gz', compression='gzip')
    ID_columns = ['Molecule ID', 'Duplicate ID', 'SMSSF ID', 'Batch Name', 
                  'Library ID', 'Plate Name', 'Plate Well', 'Run Date']
    feature_columns = ['CDD SMILES', 'rdkit SMILES', '{} MorganFP Radius {}'.format(FP_size, FP_radius)]
    label_columns = ['PriA-SSB AS % inhibition', 'PriA-SSB AS Activity', 
                     'Primary Filter', 'Retest Filter', 'PAINS Filter']

    # define unique identifiers for each row
    uniqueness_cols = ['SMSSF ID', 'Plate Name', 'Plate Well', 'Run Date', 'PriA-SSB AS % inhibition']

    # step 0: remove molecules with % inhibition <= -100.0
    concat_df = concat_df[concat_df['PriA-SSB AS % inhibition'] > lb_inh_threshold]
    
    # step 1: remove NaNs
    concat_df = concat_df[~pd.isna(concat_df['PriA-SSB AS % inhibition'])]
    concat_df = concat_df[~pd.isna(concat_df['CDD SMILES'])]
    
    # step 2: assert that there are no duplicates on the uniqueness columns 
    assert concat_df[concat_df.duplicated(subset=uniqueness_cols)].shape[0] == 0

    # step 3: add rdkit SMILES and fingerprints. Note salts are removed
    saltRemover = SaltRemover(defnFilename=rawdata_dir+'/Salts.txt')
    rdkit_mols = concat_df[feature_columns[0]].astype(str).apply((lambda x: Chem.MolFromSmiles(x)))
    rdkit_mols = rdkit_mols.apply((lambda x: saltRemover.StripMol(x)))
    concat_df[feature_columns[1]] = rdkit_mols.apply((lambda x: Chem.MolToSmiles(x)))
    concat_df[feature_columns[2]] = rdkit_mols.apply((lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 
                                                                                           radius=FP_radius, 
                                                                                           nBits=FP_size).ToBitString()))

    # step 4: add Molecule ID and Duplicate ID placeholders
    concat_df[ID_columns[0]] = list(range(concat_df.shape[0]))
    concat_df[ID_columns[1]] = 0

    # step 5: group molecules that have the same SMSSF ID under same molecule id and increasing duplicate id.
    same_smiles = concat_df.duplicated(subset='rdkit SMILES')
    same_smssfid = concat_df.duplicated(subset='SMSSF ID')
    cond_df = concat_df[same_smssfid | same_smiles]
    while cond_df.shape[0] > 0:
        smssf_id, smiles = cond_df[['SMSSF ID', 'rdkit SMILES']].iloc[0,:]
        selection_cond = (concat_df['SMSSF ID'] == smssf_id) | (concat_df['rdkit SMILES'] == smiles)
        dup_df = concat_df[selection_cond]
        mol_id = dup_df['Molecule ID'].iloc[0]
        dup_id = list(range(dup_df.shape[0]))
        
        concat_df.loc[selection_cond, 'Molecule ID'] = mol_id
        concat_df.loc[selection_cond, 'Duplicate ID'] = dup_id
        
        cond_df = cond_df[(cond_df['SMSSF ID'] != smssf_id) & (cond_df['rdkit SMILES'] != smiles)]
        
    # step 6: generate binary label according to binarization rules
    # setup and run PAINS filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    pains_catalog = FilterCatalog(params)

    concat_df['Primary Filter'] = -1.0
    concat_df['Retest Filter'] = -1.0
    concat_df['PAINS Filter'] = -1.0
    concat_df['PriA-SSB AS Activity'] = -1.0

    for mol_id in pd.unique(concat_df['Molecule ID']):
        print(mol_id)
        same_molid = concat_df['Molecule ID'] == mol_id
        grouped_df = concat_df[same_molid]
        primary_filter, retest_filter, pains_filter = binarize_rules(grouped_df, pains_catalog, binary_threshold)
        
        concat_df.loc[same_molid, 'Primary Filter'] = primary_filter
        concat_df.loc[same_molid, 'Retest Filter'] = retest_filter
        concat_df.loc[same_molid, 'PAINS Filter'] = pains_filter
        concat_df.loc[same_molid, 'PriA-SSB AS Activity'] = primary_filter and retest_filter and pains_filter
        
    # finally save the master_df
    master_df = concat_df[ID_columns+feature_columns+label_columns]
    master_df.to_csv(output_dir+'/master_mlpcn_lc.csv.gz', index=False, compression='gzip')