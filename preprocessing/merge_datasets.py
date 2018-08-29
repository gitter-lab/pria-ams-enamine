"""
    Merges MLCPN+LC raw datasets into one large master dataframe.
    Refer to "Merging MLCPN and LC datasets - description.txt" for description.
    
    Usage:
        python merge_datasets.py \
        --rawdata_dir=../datasets/raw/ \
        --output_dir=../datasets/master/ \
        [--FP_size=1024] \
        [--FP_radius=2]
"""

import argparse
import pandas as pd
import numpy as np
import os
import gzip
import time
from rdkit import Chem
from rdkit.Chem import AllChem

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--rawdata_dir', action="store", dest="rawdata_dir", required=True)
parser.add_argument('--output_dir', action="store", dest="output_dir", required=True)
parser.add_argument('--FP_size', type=int, default=1024, action="store", dest="FP_size", required=False)
parser.add_argument('--FP_radius', type=int, default=2, action="store", dest="FP_radius", required=False)

given_args = parser.parse_args()
rawdata_dir = given_args.rawdata_dir
output_dir = given_args.output_dir
FP_size = given_args.FP_size
FP_radius = given_args.FP_radius
  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = open(output_dir+'/std.out','w')

# specify raw files
raw_files = [rawdata_dir+'/lifechem123_cleaned_2017_03_10.smi',
             rawdata_dir+'/screening_smsf_continuous_2017_03_10.xlsx',
             rawdata_dir+'/lifechem4_cleaned_2017_03_10.smi',
             rawdata_dir+'/Keck_LC4_export.xlsx',
             rawdata_dir+'/pria_lc4_retest_may18.csv',
             rawdata_dir+'/keck_PriA_MLPCN.csv',
             rawdata_dir+'/keck_retest_continuous.xlsx']

# master dataframe columns
ID_columns = ['Molecule ID', 'Duplicate ID', 'SMSSF ID', 'Supplier ID']
feature_columns = ['SMILES', '{} MorganFP Radius {}'.format(FP_size, FP_radius)]
label_columns = ['PriA-SSB AS normalized % inhibition', 'PriA-SSB AS Activity']

# preallocate master dataframe
preallocate_size = 1000000
master_df = pd.DataFrame({ID_columns[0]: pd.Series([-1]*preallocate_size, dtype=np.int64),
                          ID_columns[1]: pd.Series([-1]*preallocate_size, dtype=np.int64),
                          ID_columns[2]: pd.Series(['']*preallocate_size, dtype='object'),
                          ID_columns[3]: pd.Series(['']*preallocate_size, dtype='str'),
                          feature_columns[0]: pd.Series(['']*preallocate_size, dtype='object'),
                          feature_columns[1]: pd.Series(['']*preallocate_size, dtype='str'),
                          label_columns[0]: pd.Series([np.nan]*preallocate_size, dtype='object'),
                          label_columns[1]: pd.Series([-1]*preallocate_size, dtype=np.int8)
                         },
                         columns=ID_columns+feature_columns+label_columns)

curr_molid = 0
curr_rowidx = 0

# process LC123
# process smi file
print('Processing LC123', file=output_file)
start_time = time.time()
with open(raw_files[0], 'r') as in_file:
    for line in in_file:
        curr_df = master_df.iloc[:curr_rowidx+1,:]
        
        smiles, mol_name = line.split()
        mol = Chem.MolFromSmiles(smiles)
        fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_radius, nBits=FP_size).ToBitString()
        canon_smiles = Chem.MolToSmiles(mol)
        
        if 'smssf' in mol_name.lower():
            same_smssfid = curr_df['SMSSF ID'].str.match(mol_name)
            same_smiles = curr_df['SMILES'].str.match(canon_smiles)
            
            # check if same smssfid or same smiles, treat as same molecule
            if same_smssfid.any():
                curr_df = curr_df[same_smssfid].sort_values('Duplicate ID')
                ref_molid, ref_dupid = curr_df.iloc[-1,[0,1]]
            elif same_smiles.any():
                curr_df = curr_df[same_smiles].sort_values('Duplicate ID')
                ref_molid, ref_dupid = curr_df.iloc[-1,[0,1]]
            else: # new molecule
                ref_molid, ref_dupid = curr_molid, 0
                curr_molid += 1
            master_df.loc[curr_rowidx, ['Molecule ID', 'Duplicate ID', 
                                        'SMSSF ID', 'SMILES', 
                                        '{} MorganFP Radius {}'.format(FP_size, FP_radius)]] = [ref_molid, 
                                                                                                ref_dupid+1, 
                                                                                                mol_name,
                                                                                                canon_smiles,
                                                                                                fps]
            curr_rowidx += 1

print(raw_files[0], 'compound count: {}.'.format(curr_rowidx), 'Processing time: {}'.format(time.time()-start_time), file=output_file)
print('Master DF current size: {}'.format(curr_rowidx), file=output_file)

# process main label file
start_time = time.time()
lc123_main = pd.read_excel(raw_files[1], sheet_name='Keck_Pria_Primary')
lc123_main.columns = ['SMSSF ID', 'PriA-SSB AS normalized % inhibition']
# check that smssfid is not duplicated within lc123
assert not master_df.duplicated(subset='SMSSF ID').any()
# update continuous activity score
master_df = master_df.merge(lc123_main, on='SMSSF ID', how='left', suffixes=['_x', ''])[ID_columns+feature_columns+label_columns]
print(raw_files[1], 'compound count: {}'.format(lc123_main.shape), 'Processing time: {}'.format(time.time()-start_time), file=output_file)
del lc123_main
                  
# process retest data
start_time = time.time()
lc123_retest = pd.read_excel(raw_files[6], sheet_name='LC123')
lc123_retest.columns = ['SMSSF ID', 'AS_PRIMARY', 'AS_RETEST', 'FP_RETEST']
lc123_retest = lc123_retest[['SMSSF ID', 'AS_PRIMARY', 'AS_RETEST']]
for idx, row in lc123_retest.iterrows():
    curr_df = master_df.iloc[:curr_rowidx+1,:]
    mol_name, activity_score_primary, activity_score_retest = lc123_retest.iloc[idx,:]

    if 'smssf' in mol_name.lower():
        same_smssfid = curr_df['SMSSF ID'].str.match(mol_name)

        # check if same smssfid treat as same molecule
        if same_smssfid.any():
            curr_df = curr_df[same_smssfid].sort_values('Duplicate ID')
            ref_molid, ref_dupid, suppid, canon_smiles, fps = curr_df.iloc[-1,[0, 1, 3, 4, 5]]
            
            master_df.loc[curr_rowidx, ['Molecule ID', 'Duplicate ID', 
                                        'SMSSF ID', 'SMILES', 
                                        '{} MorganFP Radius {}'.format(FP_size, FP_radius),
                                        'Supplier ID', 
                                        'PriA-SSB AS normalized % inhibition']] = [ref_molid, ref_dupid+1, 
                                                                                   mol_name, canon_smiles,
                                                                                   fps, suppid, activity_score_retest]
            curr_rowidx += 1
        else: # do nothing as we don't have smile info for this retest
            pass

print(raw_files[6], 'compound count: {}'.format(lc123_retest.shape), 'Processing time: {}'.format(time.time()-start_time), file=output_file)
print('Master DF current size: {}'.format(curr_rowidx), file=output_file)
del lc123_retest

# process LC4
print('\nProcessing LC4', file=output_file)
start_time = time.time()
lc4_df1 = pd.read_excel(raw_files[3], sheet_name='LifeChem4 Export')
lc4_df1.columns = ['SMSSF ID', 'Supplier ID', 'PriA-SSB AS']
print(raw_files[3], 'compound count: {}'.format(lc4_df1.shape), file=output_file)

lc4_df2 = pd.read_csv(raw_files[4])
lc4_df2.columns = ['SMSSF ID', 'Primary', 'Retest', 'Active']
print(raw_files[4], 'compound count: {}'.format(lc4_df2.shape), file=output_file)

lc4_merged_df = lc4_df1.copy()

temp_df = lc4_df2[['SMSSF ID', 'Primary']].merge(lc4_df1, on='SMSSF ID', how='inner')
temp_df = temp_df[['SMSSF ID', 'Supplier ID', 'Primary']]
temp_df.columns = ['SMSSF ID', 'Supplier ID', 'PriA-SSB AS']
lc4_merged_df = lc4_merged_df.append(temp_df, ignore_index=True)

temp_df = lc4_df2[['SMSSF ID', 'Retest']].merge(lc4_df1, on='SMSSF ID', how='inner')
temp_df = temp_df[['SMSSF ID', 'Supplier ID', 'Retest']]
temp_df.columns = ['SMSSF ID', 'Supplier ID', 'PriA-SSB AS']
lc4_merged_df = lc4_merged_df.append(temp_df, ignore_index=True)
lc4_merged_df = lc4_merged_df.drop_duplicates(subset=['SMSSF ID', 'PriA-SSB AS'])

smiles_df = pd.read_csv(raw_files[2], delimiter=' ', header=None)
smiles_df.columns = ['SMILES', 'Supplier ID']
print(raw_files[2], 'compound count: {}'.format(smiles_df.shape), file=output_file)

lc4_merged_df = lc4_merged_df.merge(smiles_df, on='Supplier ID', how='left')
lc4_merged_df = lc4_merged_df[~lc4_merged_df.SMILES.isna()]
lc4_merged_df = lc4_merged_df[['SMSSF ID', 'Supplier ID', 'SMILES', 'PriA-SSB AS']]

print('LC4 merged compound count: {}'.format(lc4_merged_df.shape), file=output_file)
for idx, row in lc4_merged_df.iterrows():
    curr_df = master_df.iloc[:curr_rowidx+1,:]

    mol_name, suppid, smiles, activity_score = lc4_merged_df.iloc[idx,:]
    mol = Chem.MolFromSmiles(smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_radius, nBits=FP_size).ToBitString()
    canon_smiles = Chem.MolToSmiles(mol)

    if 'smssf' in mol_name.lower():
        same_smssfid = curr_df['SMSSF ID'].str.match(mol_name)
        same_smiles = curr_df['SMILES'].str.match(canon_smiles)

        # check if same smssfid or same smiles, treat as same molecule
        if same_smssfid.any():
            curr_df = curr_df[same_smssfid].sort_values('Duplicate ID')
            ref_molid, ref_dupid = curr_df.iloc[-1,[0,1]]
        elif same_smiles.any():
            curr_df = curr_df[same_smiles].sort_values('Duplicate ID')
            ref_molid, ref_dupid = curr_df.iloc[-1,[0,1]]
        else: # new molecule
            ref_molid, ref_dupid = curr_molid, 0
            curr_molid += 1
        master_df.loc[curr_rowidx, ['Molecule ID', 'Duplicate ID', 
                                    'SMSSF ID', 'SMILES', 
                                    '{} MorganFP Radius {}'.format(FP_size, FP_radius),
                                    'Supplier ID', 
                                    'PriA-SSB AS normalized % inhibition']] = [ref_molid, ref_dupid+1, 
                                                                               mol_name, canon_smiles,
                                                                               fps, suppid, activity_score]
        curr_rowidx += 1

print('LC4 merged compound count: {}'.format(lc4_merged_df.shape), 'Processing time: {}'.format(time.time()-start_time), file=output_file)
print('Master DF current size: {}'.format(curr_rowidx), file=output_file)
del lc4_df1, lc4_df2, lc4_merged_df, smiles_df, temp_df
 
# process MLPCN
print('\nProcessing MLPCN', file=output_file)
# main MLPCN file
start_time = time.time()
mlpcn_df = pd.read_csv(raw_files[5])
mlpcn_df.columns = ['SMSSF ID', 'SMILES', 'Supplier ID', 'PID', 'PriA-SSB AS', 'Activity']
mlpcn_df = mlpcn_df[['SMSSF ID', 'Supplier ID', 'SMILES', 'PriA-SSB AS']]

for idx, row in mlpcn_df.iterrows():
    curr_df = master_df.iloc[:curr_rowidx+1,:]

    mol_name, suppid, smiles, activity_score = mlpcn_df.iloc[idx,:]
    mol = Chem.MolFromSmiles(smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_radius, nBits=FP_size).ToBitString()
    canon_smiles = Chem.MolToSmiles(mol)

    if 'smssf' in mol_name.lower():
        same_smssfid = curr_df['SMSSF ID'].str.match(mol_name)
        same_smiles = curr_df['SMILES'].str.match(canon_smiles)

        # check if same smssfid or same smiles, treat as same molecule
        if same_smssfid.any():
            curr_df = curr_df[same_smssfid].sort_values('Duplicate ID')
            ref_molid, ref_dupid = curr_df.iloc[-1,[0,1]]
        elif same_smiles.any():
            curr_df = curr_df[same_smiles].sort_values('Duplicate ID')
            ref_molid, ref_dupid = curr_df.iloc[-1,[0,1]]
        else: # new molecule
            ref_molid, ref_dupid = curr_molid, 0
            curr_molid += 1
        master_df.loc[curr_rowidx, ['Molecule ID', 'Duplicate ID', 
                                    'SMSSF ID', 'SMILES', 
                                    '{} MorganFP Radius {}'.format(FP_size, FP_radius),
                                    'Supplier ID', 
                                    'PriA-SSB AS normalized % inhibition']] = [ref_molid, ref_dupid+1, 
                                                                               mol_name, canon_smiles,
                                                                               fps, suppid, activity_score]
        curr_rowidx += 1

print(raw_files[5], 'compound count: {}'.format(mlpcn_df.shape), 'Processing time: {}'.format(time.time()-start_time), file=output_file)
print('Master DF current size: {}'.format(curr_rowidx), file=output_file)
del mlpcn_df

# process retest data
start_time = time.time()
mlpcn_retest = pd.read_excel(raw_files[6], sheet_name='MLPCN')
mlpcn_retest.columns = ['SMSSF ID', 'AS_PRIMARY', 'AS_RETEST', 'FP_RETEST']
mlpcn_retest = mlpcn_retest[['SMSSF ID', 'AS_PRIMARY', 'AS_RETEST']]
for idx, row in mlpcn_retest.iterrows():
    curr_df = master_df.iloc[:curr_rowidx+1,:]

    mol_name, activity_score_primary, activity_score_retest = mlpcn_retest.iloc[idx,:]

    if 'smssf' in mol_name.lower():
        same_smssfid = curr_df['SMSSF ID'].str.match(mol_name)

        # check if same smssfid treat as same molecule
        if same_smssfid.any():
            curr_df = curr_df[same_smssfid].sort_values('Duplicate ID')
            ref_molid, ref_dupid, suppid, canon_smiles, fps = curr_df.iloc[-1,[0, 1, 3, 4, 5]]
            
            master_df.loc[curr_rowidx, ['Molecule ID', 'Duplicate ID', 
                                        'SMSSF ID', 'SMILES', 
                                        '{} MorganFP Radius {}'.format(FP_size, FP_radius),
                                        'Supplier ID', 
                                        'PriA-SSB AS normalized % inhibition']] = [ref_molid, ref_dupid+1, 
                                                                                   mol_name, canon_smiles,
                                                                                   fps, suppid, activity_score_retest]
            curr_rowidx += 1
            print(curr_rowidx, file=output_file)
        else: # do nothing as we don't have smile info for this retest
            pass

print(raw_files[6], 'compound count: {}'.format(mlpcn_retest.shape), 'Processing time: {}'.format(time.time()-start_time), file=output_file)
print('Master DF current size: {}'.format(curr_rowidx), file=output_file)
del mlpcn_retest

# done processing raw files now remove NaN rows
master_df = master_df[master_df['Molecule ID'] != -1]
master_df = master_df.sort_values(['Molecule ID', 'Duplicate ID'], ascending=[True, True])

# deduplicate entries based on SMSSF ID, SMILES, and Activity Score for double measure
print('Master DF BEFORE deduplication: {}'.format(master_df.shape), file=output_file)
master_df = master_df.drop_duplicates(subset=['SMSSF ID', 'SMILES', 'PriA-SSB AS normalized % inhibition'])
print('Master DF AFTER deduplication on SMSSF ID, SMILES, and Activity Score: {}'.format(master_df.shape), file=output_file)

output_file.close()
master_df.to_csv(output_dir+'/master_mlpcn_lc.csv', ignore_index=True)