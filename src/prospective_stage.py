from __future__ import print_function

import json
import argparse
import numpy as np
import pandas as pd
from function import *
import glob
from random_forest_classification import RandomForestClassification
from baseline_similarity import SimilarityBaseline
    
def train(model):
    K = 10
    directory = '../datasets/keck_pria/fold_{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list = np.array(file_list)

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    train_index = np.arange(10)
    train_file_list = file_list[train_index]
    print('train files ', train_file_list)
    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    print('done data preparation')

    print('X_train\t', X_train.shape)
    print('y_train\t', y_train.shape)
    
    if model == 'random_forest_classification':
        task = RandomForestClassification(conf=conf)
    elif model == 'baseline':
        task = SimilarityBaseline(conf=conf)
    task.train_and_predict(X_train, y_train, None, None, weight_file)
    return
    
def prospective_baseline():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    task = SimilarityBaseline(conf=conf)
    
    handler =  open('../output/final_stage/{}_{}_prediction.out'.format(target, 'baseline'), 'w')
    
    for count in range(50):
        df = pd.read_csv('../datasets/{}/{}.csv.gz'.format(target, count))
        print('{}\tshape:{}'.format(count, df.shape))
        
        smiles_list = df['smiles'].tolist()
        fingerprints_list = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in df['fingerprints']]).astype(float)

        pred_values = task.predict_with_existing(fingerprints_list, weight_file)[:,0]
        print('shape: {},\tfirst 10 values: {}'.format(pred_values.shape, pred_values[:10]))
        print('over 0.1 has {}/{}'.format(sum(pred_values > 0.1), len(pred_values)))

        for smiles, pred_value in zip(smiles_list, pred_values):
            print('{}\t{}'.format(smiles, pred_value), file=handler)
        print()
    return
    
def prospective_rf():
    with open(config_json_file, 'r') as f:
        task_conf = json.load(f)
    random_forest_model = RandomForestClassification(conf=task_conf)
    model = random_forest_model.load_model(weight_file)

    handler = open('../output/final_stage/{}_{}_prediction.out'.format(target, 'random_forest_classification'), 'w')

    for count in range(50):
        df = pd.read_csv('../datasets/{}/{}.csv.gz'.format(target, count))
        print('{}\tshape:{}'.format(count, df.shape))

        old_smiles_list = df['old smiles'].tolist()
        smiles_list = df['smiles'].tolist()
        fingerprints_list = df['fingerprints'].tolist()
        id_list = df['datestamp'].tolist()
        fingerprints_list = map(lambda x: list(x), fingerprints_list)
        fingerprints_list = np.array(fingerprints_list)
        fingerprints_list= fingerprints_list.astype(float)
        print(fingerprints_list.shape)

        pred_values = model.predict_proba(fingerprints_list)[:, 1]
        print('shape: {},\tfirst 10 values: {}'.format(pred_values.shape, pred_values[:10]))
        print('over 0.1 has {}/{}'.format(sum(pred_values > 0.1), len(pred_values)))

        for ori_smiles, smiles, id, pred_value in zip(old_smiles_list, smiles_list, id_list, pred_values):
            print('{}\t{}\t{}\t{}'.format(ori_smiles, smiles, id, pred_value), file=handler)
        print()
    return
    
"""
Example usage:
python prospective_stage.py \
        --config_json_file=../config/random_forest_classification/139.json \
        --weight_file=../model_weight/final_stage/random_forest_classification_139.pkl \
        --model=random_forest_classification \
        --mode=prediction
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', dest="config_json_file",
                        action="store", required=True)
    parser.add_argument('--weight_file', dest="weight_file",
                        action="store", required=True)
    parser.add_argument('--model', dest='model',
                        action='store', required=True)
    parser.add_argument('--mode', dest='mode',
                        action='store', required=True)
    parser.add_argument('--target', default='aldrich',
                        action="store", required=True)
    given_args = parser.parse_args()
    
    config_json_file = given_args.config_json_file
    train_file_dir_fmt = given_args.train_file_dir_fmt
    weight_file = given_args.weight_file
    model = given_args.model
    mode = given_args.mode
    target = given_args.target
    
    if model not in ['random_forest_classification', 'baseline']:
        raise Exception('model should be among [{}, {}].'.format(
            'random_forest_classification',
            'baseline'
            ))
     if mode not in ['training', 'prediction']:
        raise Exception('mode should be among [{}, {}].'.format(
            'training',
            'prediction'
            ))
            
    if mode == 'training':
        print('Start Training.')
        train(model)
    elif mode == 'prediction':
        print('Start Predicting.')
        if model == 'random_forest_classification':
            prospective_rf()
        elif model == 'baseline':
            prospective_baseline()
