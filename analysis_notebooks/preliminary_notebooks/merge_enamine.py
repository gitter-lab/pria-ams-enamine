"""
    Usage:
        python aldd_compute_reqs.py \
        --log_file_name=../logs/aldd.log \
        --job_name=HS_ClusterBasedRandom_TASK_pcba-aid602332_BATCH_0_START_1_ITERS_0 \
        --condor_subtemplate=./aldd_template.sub \
        --new_condor_subname=./HS_ClusterBasedRandom_TASK_pcba-aid602332_BATCH_0_START_1_ITERS_1.sub
"""

import argparse

if __name__ ==  '__main__':
    import pandas as pd
    import numpy as np
    import glob

    train_df = pd.read_csv('../datasets/folds/training_df_single_fold_with_clustering.csv.gz')
    top_df = pd.read_csv('../datasets/real_top_10000.csv.gz')
    ams_df = pd.read_csv('../datasets/ams_order_results.csv.gz')
    clustering = pd.read_csv('../datasets/real_clustering.csv.gz')

    real_costs_1 = pd.read_csv('../datasets/real_costs_2020_04_09.csv')
    real_costs_2 = pd.read_csv('../datasets/premium_1536_costs_2020_04_09.csv')

    print('real_costs: {}'.format(real_costs_1.shape))
    print('real_costs_2: {}'.format(real_costs_2.shape))

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.SaltRemover import SaltRemover
    from rdkit.Chem.FilterCatalog import *

    saltRemover = SaltRemover(defnFilename='../datasets/raw/Salts.txt')
    rdkit_mols = real_costs_1['SMILES'].astype(str).apply((lambda x: Chem.MolFromSmiles(x)))
    rdkit_mols = rdkit_mols.apply((lambda x: saltRemover.StripMol(x)))
    smiles = rdkit_mols.apply((lambda x: Chem.MolToSmiles(x)))
    real_costs_1['rdkit SMILES'] = smiles

    rdkit_mols = real_costs_2['SMILES'].astype(str).apply((lambda x: Chem.MolFromSmiles(x)))
    rdkit_mols = rdkit_mols.apply((lambda x: saltRemover.StripMol(x)))
    smiles = rdkit_mols.apply((lambda x: Chem.MolToSmiles(x)))
    real_costs_2['rdkit SMILES'] = smiles

    real_costs = pd.concat([real_costs_1, real_costs_2])
    merge_smile = pd.merge(top_df, real_costs, left_on='rdkit SMILES', right_on='rdkit SMILES', suffixes=['', '_y'])
    merge_id = pd.merge(top_df, real_costs, left_on='idnumber', right_on='ID Enamine', suffixes=['', '_y'])

    print(merge_smile.shape, merge_id.shape, merge_id[merge_id['idnumber'].isin(merge_smile['idnumber'])].shape)

    print(np.setdiff1d(merge_id['idnumber'], merge_smile['idnumber']))

    # make note of racemic cpds
    racemic = real_costs_2[~real_costs_2['ID Enamine'].isin(merge_smile['idnumber'])]

    merge_smile = merge_smile.drop(['SMILES', 'MW_y'], axis=1)
    top_real_clustering = merge_smile

    ams_clusters = clustering[clustering['dataset'] == 'ams']
    train_clusters = clustering[clustering['dataset'] == 'train']
    tr_clusters = top_real_clustering['TB_0.4 ID'].values

    exists_in_ams = np.intersect1d(tr_clusters, ams_clusters['TB_0.4 ID'].values)
    ams_active_clusters = exists_in_ams[ams_clusters.groupby('TB_0.4 ID').sum()['Hit'].loc[exists_in_ams] > 0]
    exists_in_train = np.intersect1d(tr_clusters, train_clusters['TB_0.4 ID'].values)
    train_active_clusters = exists_in_train[train_clusters.groupby('TB_0.4 ID').sum()['Hit'].loc[exists_in_train] > 0]

    top_real_clustering['Is Train cluster?'] = 0
    top_real_clustering['Is Train active cluster?'] = 0
    top_real_clustering['Is AMS cluster?'] = 0
    top_real_clustering['Is AMS active cluster?'] = 0

    top_real_clustering.loc[top_real_clustering['TB_0.4 ID'].isin(exists_in_train),'Is Train cluster?'] = 1
    top_real_clustering.loc[top_real_clustering['TB_0.4 ID'].isin(train_active_clusters),'Is Train active cluster?'] = 1
    top_real_clustering.loc[top_real_clustering['TB_0.4 ID'].isin(exists_in_ams),'Is AMS cluster?'] = 1
    top_real_clustering.loc[top_real_clustering['TB_0.4 ID'].isin(ams_active_clusters),'Is AMS active cluster?'] = 1

    print(real_costs_1.shape, top_real_clustering[top_real_clustering['idnumber'].isin(real_costs_1['ID Enamine'])].shape)
    print(real_costs_2.shape, top_real_clustering[top_real_clustering['idnumber'].isin(real_costs_2['ID Enamine'])].shape)
    
    # add racemic
    racemic['idnumber'] = racemic['ID Enamine']
    FP_radius=2
    FP_size=1024
    racemic = real_costs_2[~real_costs_2['ID Enamine'].isin(merge_smile['idnumber'])]
    rdkit_mols = racemic['SMILES'].astype(str).apply((lambda x: Chem.MolFromSmiles(x)))
    rdkit_mols = rdkit_mols.apply((lambda x: saltRemover.StripMol(x)))
    racemic['1024 MorganFP Radius 2'] = rdkit_mols.apply((lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 
                                                                                           radius=FP_radius, 
                                                                                           nBits=FP_size).ToBitString()))
    racemic['REAL SMILES'] = racemic['SMILES']