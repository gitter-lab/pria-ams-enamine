"""
    Processes REAL database by predicting activity using random forest model. 
    This file processes a REAL db file in chunks based on process_id.
    
    Usage:
        python predict_real_db.py \
        --rf_model_file=./random_forest_classification_139.pkl \
        --real_db_file=../../../REAL_db/2019q1-2_Enamine_REAL_723M_SMILES_Part_01.smiles \
        --output_csv_file=rf_preds_part_01_process_0.csv \
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
    parser.add_argument('--output_csv_file', action="store", dest="output_csv_file", required=True)
    parser.add_argument('--process_id', type=int, action="store", dest="process_id", required=True)
    parser.add_argument('--instances_per_process', type=int, default=524288, action="store", dest="instances_per_process", required=False)
    parser.add_argument('--chunksize', type=int, default=65536, action="store", dest="chunksize", required=False)
    
    given_args = parser.parse_args()
    rf_model_file = given_args.rf_model_file
    real_db_file = given_args.real_db_file
    output_csv_file = given_args.output_csv_file
    process_id = given_args.process_id
    instances_per_process = given_args.instances_per_process
    chunksize = given_args.chunksize
    
    # load random forest model
    with open('./139.json', 'r') as f:
        task_conf = json.load(f)
    rf_model = RandomForestClassification(conf=task_conf)
    rf_model = rf_model.load_model(rf_model_file)
    pathlib.Path(output_csv_file).parent.mkdir(parents=True, exist_ok=True)
                                
    # process real db file in chunks, generating features, and predicting using random forest model.
    saltRemover = SaltRemover(defnFilename='../datasets/raw/Salts.txt')
    FP_radius, FP_size = 2, 1024
    n_cpds = 0
    data_df = pd.read_csv(real_db_file, chunksize=chunksize, 
                          usecols=[0], header=None, skiprows=1+process_id*instances_per_process, 
                          nrows=instances_per_process, delimiter='\t')
    with open(output_csv_file, 'w') as outputfile:
        outputfile.write('smiles,rf_preds\n')
        for chunk_idx, chunk_df in enumerate(data_df):
            n_cpds += chunk_df.shape[0]
            chunk_smiles = chunk_df[0]
            rdkit_mols = chunk_smiles.astype(str).apply((lambda x: Chem.MolFromSmiles(x)))
            rdkit_mols = rdkit_mols.apply((lambda x: saltRemover.StripMol(x)))
            X = rdkit_mols.apply((lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 
                                                                                  radius=FP_radius, 
                                                                                  nBits=FP_size).ToBitString()))
            X = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in X]).astype(float) # this is from: https://stackoverflow.com/a/29091970
            chunk_preds = rf_model.predict_proba(X)[:,1]
            
            for smile, pred in zip(chunk_smiles.tolist(), chunk_preds):
                outputfile.write('{},{}\n'.format(smile, pred))
        
    end_time = time.time()
    total_time_minutes = (end_time-start_time)/60.0
    print('Total time: {} minutes to process {} cpds.'.format(total_time_minutes, n_cpds))