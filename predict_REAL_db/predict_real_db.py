"""
    Processes REAL database by predicting activity using random forest model. 
    This file processes a REAL db file in chunks based on process_id.
    
    Usage:
        python predict_real_db.py \
        --rf_model_file=./random_forest_classification_139.pkl \
        --real_db_file=../../../REAL_db/2019q1-2_Enamine_REAL_723M_SMILES_Part_01.smiles \
        --output_dir=./ \
        --output_memmap_file_fmt=rf_preds_part_01_process_{}_ncpds_{}.dat \
        --process_id=0 \ 
        [--instances_per_process=524288] \
        [--chunksize=65536]
"""

import argparse
import pandas as pd
import numpy as np
import pathlib
import gzip
import time
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.FilterCatalog import *

import sys
sys.path.insert(0,'../src/')
from random_forest_classification import RandomForestClassification
    
if __name__ == '__main__':
    start_time = time.time()
    # read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_model_file', action="store", dest="rf_model_file", required=True)
    parser.add_argument('--real_db_file', action="store", dest="real_db_file", required=True)
    parser.add_argument('--output_dir', action="store", dest="output_dir", required=True)
    parser.add_argument('--output_memmap_file_fmt', action="store", dest="output_memmap_file_fmt", required=True)
    parser.add_argument('--process_id', type=int, action="store", dest="process_id", required=True)
    parser.add_argument('--instances_per_process', type=int, default=524288, action="store", dest="instances_per_process", required=False)
    parser.add_argument('--chunksize', type=int, default=65536, action="store", dest="chunksize", required=False)
    
    given_args = parser.parse_args()
    rf_model_file = given_args.rf_model_file
    real_db_file = given_args.real_db_file
    output_dir = given_args.output_dir
    output_memmap_file_fmt = given_args.output_memmap_file_fmt
    process_id = given_args.process_id
    instances_per_process = given_args.instances_per_process
    chunksize = given_args.chunksize
    
    with open('./139.json', 'r') as f:
        task_conf = json.load(f)
    rf_model = RandomForestClassification(conf=task_conf)
    rf_model = rf_model.load_model(rf_model_file)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_memmap_file = output_dir+'/'+output_memmap_file_fmt.format(process_id, instances_per_process)
    rf_preds = np.memmap(output_memmap_file, 
                         dtype='float16', mode='w+', shape=(instances_per_process,2))
                                
    # process real db file in chunks, generating features, and predicting using random forest model.
    saltRemover = SaltRemover(defnFilename='../datasets/raw/Salts.txt')
    FP_radius, FP_size = 2, 1024
    n_cpds = 0
    for chunk_idx, chunk_df in enumerate(pd.read_csv(real_db_file, chunksize=chunksize, 
                                                     usecols=[0], header=None, skiprows=1+process_id*instances_per_process, 
                                                     nrows=instances_per_process, delimiter='\t')):
        n_cpds += chunk_df.shape[0]                                   
        rdkit_mols = chunk_df[0].astype(str).apply((lambda x: Chem.MolFromSmiles(x)))
        rdkit_mols = rdkit_mols.apply((lambda x: saltRemover.StripMol(x)))
        X = rdkit_mols.apply((lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 
                                                                              radius=FP_radius, 
                                                                              nBits=FP_size).ToBitString()))
        X = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in X]).astype(float) # this is from: https://stackoverflow.com/a/29091970
        chunk_preds = rf_model.predict_proba(X)[:,1]
        rf_preds[chunk_idx*chunksize:(chunk_idx+1)*chunksize,0] = chunk_preds
        rf_preds[chunk_idx*chunksize:(chunk_idx+1)*chunksize,1] = np.arange(chunk_idx*chunksize, (chunk_idx+1)*chunksize)
        
    del rf_preds
    end_time = time.time()
    total_time_minutes = (end_time-start_time)/60.0
    print('Total time: {} minutes to process {} cpds.'.format(total_time_minutes, n_cpds))