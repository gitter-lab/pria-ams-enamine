from __future__ import print_function

import json
import argparse
import numpy as np
import pandas as pd
from function import *
import glob

# specify dataset
K = 10
directory = '../datasets/keck_pria/fold_{}.csv'
file_list = []
for i in range(K):
    file_list.append(directory.format(i))
file_list = np.array(file_list)

def prospective_baseline():
    from baseline_similarity import SimilarityBaseline

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    task = SimilarityBaseline(conf=conf)
    
    for target in ['aldrich']:
        handler = open(prospective_preds_file.format(target, 'baseline'), 'w')
        
        for count in range(50):
            df = pd.read_csv('../datasets/{}/{}.csv.gz'.format(target, count))
            print('{}\tshape:{}'.format(count, df.shape))
            
            smiles_list = df['smiles'].tolist()
            fingerprints_list = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in df['fingerprints']]).astype(float)

            pred_values = task.predict_with_existing(fingerprints_list[:20], weight_file)[:,0]
            print('shape: {},\tfirst 10 values: {}'.format(pred_values.shape, pred_values[:10]))
            print('over 0.1 has {}/{}'.format(sum(pred_values > 0.1), len(pred_values)))

            for smiles, pred_value in zip(smiles_list, pred_values):
                print('{}\t{}'.format(smiles, pred_value), file=handler)
            print()
    

"""
Example usage:
python prospective_keck.py \
        --config_json_file=../config/baseline_similarity.json \
        --train_file_dir_fmt=../datasets/keck_pria/fold_*.csv \
        --weight_file=baseline_weight.npy \
        --prospective_preds_file=baseline_aldrich_preds.npz \
        --model=baseline
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', dest="config_json_file",
                        action="store", required=True)
    parser.add_argument('--train_file_dir_fmt', dest="train_file_dir_fmt",
                        action="store", required=True)
    parser.add_argument('--weight_file', dest="weight_file",
                        action="store", required=True)
    parser.add_argument('--prospective_preds_file', dest="prospective_preds_file",
                        action="store", required=True)
    parser.add_argument('--model', dest='model',
                        action='store', required=True)
    given_args = parser.parse_args()

    config_json_file = given_args.config_json_file
    train_file_dir_fmt = given_args.train_file_dir_fmt
    weight_file = given_args.weight_file
    prospective_preds_file = given_args.prospective_preds_file
    model = given_args.model
    
    train_file_list = glob.glob(train_file_dir_fmt)
    
    if model == 'baseline':
        prospective_baseline()
