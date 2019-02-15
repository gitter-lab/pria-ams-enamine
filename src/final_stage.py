from __future__ import print_function

import json
import argparse
import numpy as np
import pandas as pd
from function import *
from xgboost_classification import XGBoostClassification


def train():
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

    val_index = [9]
    complete_index = np.arange(10)
    train_index = filter(lambda x: x not in val_index, complete_index)

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index]

    print('train files ', train_file_list)
    print('val files ', val_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='1024 MorganFP Radius 2',
                                             label_name_list=label_name_list)
    print('done data preparation')

    print('X_train\t', X_train.shape)
    print('y_train\t', y_train.shape)

    task = XGBoostClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, None, None, weight_file)
    return


def predict():
    with open(config_json_file, 'r') as f:
        task_conf = json.load(f)
    xgboost_model = XGBoostClassification(conf=task_conf)
    model = xgboost_model.load_model(weight_file)

    handler = open('../output/final_stage/{}.out'.format(target), 'w')

    for count in range(50):
        df = pd.read_csv('../datasets/{}/{}.csv.gz'.format(target, count))
        print('{}\tshape:{}'.format(count, df.shape))

        def apply(x):
            x = x.replace('[', '')
            x = x.replace(']', '')
            x = x.replace(',', '')
            x = x.replace(' ', '')
            x = x.replace('\'', '')
            return list(x)

        smiles_list = df['smiles'].tolist()
        fingerprints_list = df['fingerprints'].tolist()
        fingerprints_list = map(lambda x: apply(x), fingerprints_list)
        # print(fingerprints_list[0], len(fingerprints_list[0]))
        fingerprints_list = np.array(fingerprints_list)
        fingerprints_list= fingerprints_list.astype(float)
        print(fingerprints_list.shape)

        pred_values = model.predict_proba(fingerprints_list)[:, 1]
        print('shape: {},\tfirst 10 values: {}'.format(pred_values.shape, pred_values[:10]))
        print('over 0.1 has {}/{}'.format(sum(pred_values > 0.1), len(pred_values)))

        for smiles, pred_value in zip(smiles_list, pred_values):
            print('{}\t{}'.format(smiles, pred_value), file=handler)
        print()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='aldrich')
    parser.add_argument('--config_json_file', default='../config/xgboost_classification/140.json')
    parser.add_argument('--weight_file', default='../model_weight/final_stage/xgboost_classification_140.pkl')
    parser.add_argument('--mode', default='training')

    args = parser.parse_args()
    target = args.target
    config_json_file = args.config_json_file
    weight_file = args.weight_file
    mode = args.mode

    if mode == 'training':
        print('Start Training.')
        train()
    elif mode == 'prediction':
        print('Start Predictiong.')
        predict()
