from __future__ import print_function

import argparse
import pandas as pd
import csv
import numpy as np
import json
import sys
from function import *


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

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_test, _ = extract_feature_and_label(test_pd,
                                          feature_name='1024 MorganFP Radius 2',
                                          label_name_list=label_name_list)
    print('done data preparation')

    print('X_train\t', X_train.shape)
    print('y_train\t', y_train.shape)

    task = SimilarityBaseline(conf=conf)
    task.train_and_predict(X_train, y_train, 
                           X_val=None, y_val=None, 
                           X_test=None, y_test=None, 
                           weight_file=weight_file)
    
    y_pred_on_test = task.predict_with_existing(X_test, weight_file)
    return y_pred_on_test
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', dest="config_json_file",
                        action="store", required=True)
    parser.add_argument('--train_file_list', dest="train_file_list",
                        action="store", required=True)
    parser.add_argument('--test_file_list', dest="test_file_list",
                        action="store", required=True)
    parser.add_argument('--weight_file', dest="weight_file",
                        action="store", required=True)
    parser.add_argument('--model', dest='model',
                        action='store', required=True)
    given_args = parser.parse_args()

    config_json_file = given_args.config_json_file
    train_file_list = given_args.train_file_list
    test_file_list = given_args.test_file_list
    weight_file = given_args.weight_file
    model = given_args.model

    if model == 'baseline':
        y_pred_on_test = prospective_baseline()
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}, {}, {}, {}, {}].'.format(
            'single_deep_classification',
            'single_deep_regression',
            'multi_deep_classification',
            'random_forest_classification',
            'random_forest_regression',
            'xgboost_classification',
            'xgboost_regression',
            'ensemble'
        ))
        
    # save y_pred_on_test?
