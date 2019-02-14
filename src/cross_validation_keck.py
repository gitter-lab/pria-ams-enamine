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


def run_single_deep_classification(running_index):
    from deep_classification import SingleClassification
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound-1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    # read data
    if running_index == 5:
        test_index = [9]
        val_index = [8]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        val_index = [2 * running_index]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('val files ', val_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='1024 MorganFP Radius 2',
                                             label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)

    task = SingleClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)

    return


def run_single_deep_regression(running_index):
    from deep_regression import SingleRegression
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound - 1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    # read data
    if running_index == 5:
        test_index = [9]
        val_index = [8]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        val_index = [2 * running_index]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('val files ', val_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='1024 MorganFP Radius 2',
                                             label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)

    y_train_binary = reshape_data_into_2_dim(y_train[:, 0])
    y_train_continuous = reshape_data_into_2_dim(y_train[:, 1])
    y_val_binary = reshape_data_into_2_dim(y_val[:, 0])
    y_val_continuous = reshape_data_into_2_dim(y_val[:, 1])
    y_test_binary = reshape_data_into_2_dim(y_test[:, 0])
    y_test_continuous = reshape_data_into_2_dim(y_test[:, 1])
    print('done data preparation')

    task = SingleRegression(conf=conf)
    task.train_and_predict(X_train, y_train_continuous, y_train_binary,
                           X_val, y_val_continuous, y_val_binary,
                           X_test, y_test_continuous, y_test_binary,
                           weight_file)
    return


def run_random_forest_classification(running_index):
    from random_forest_classification import RandomForestClassification
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound - 1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    # read data
    if running_index == 5:
        test_index = [9]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index, complete_index)

    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)

    task = RandomForestClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)
    return


def run_random_forest_regression(running_index):
    from random_forest_regression import RandomForestRegression
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound - 1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    # read data
    if running_index == 5:
        test_index = [9]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index, complete_index)

    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)

    y_train_binary = reshape_data_into_2_dim(y_train[:, 0])
    y_train_continuous = reshape_data_into_2_dim(y_train[:, 1])
    y_test_binary = reshape_data_into_2_dim(y_test[:, 0])
    y_test_continuous = reshape_data_into_2_dim(y_test[:, 1])
    print('done data preparation')

    task = RandomForestRegression(conf=conf)
    task.train_and_predict(X_train, y_train_continuous, y_train_binary,
                           X_test, y_test_continuous, y_test_binary,
                           weight_file)
    return


def run_xgboost_classification(running_index):
    from xgboost_classification import XGBoostClassification
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound - 1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    # read data
    if running_index == 5:
        test_index = [9]
        val_index = [8]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        val_index = [2 * running_index]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('val files ', val_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='1024 MorganFP Radius 2',
                                             label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)
    print('done data preparation')

    print('X_train\t', X_train.shape)
    print('y_train\t', y_train.shape)

    task = XGBoostClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)
    return


def run_xgboost_regression(running_index):
    from xgboost_regression import XGBoostRegression
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound - 1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    # read data
    if running_index == 5:
        test_index = [9]
        val_index = [8]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        val_index = [2 * running_index]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('val files ', val_file_list)
    print('test files ', test_file_list)

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)
    test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='1024 MorganFP Radius 2',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='1024 MorganFP Radius 2',
                                             label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)

    y_train_binary = reshape_data_into_2_dim(y_train[:, 0])
    y_train_continuous = reshape_data_into_2_dim(y_train[:, 1])
    y_val_binary = reshape_data_into_2_dim(y_val[:, 0])
    y_val_continuous = reshape_data_into_2_dim(y_val[:, 1])
    y_test_binary = reshape_data_into_2_dim(y_test[:, 0])
    y_test_continuous = reshape_data_into_2_dim(y_test[:, 1])
    print('done data preparation')

    task = XGBoostRegression(conf=conf)
    task.train_and_predict(X_train, y_train_continuous, y_train_binary,
                           X_val, y_val_continuous, y_val_binary,
                           X_test, y_test_continuous, y_test_binary,
                           weight_file)
    return


def run_character_rnn_classification(running_index):
    from character_rnn_classification import CharacterRNNClassification
    from keras.preprocessing import sequence
    if running_index >= cross_validation_upper_bound:
        raise ValueError('Process number out of limit. At most {}.'.format(cross_validation_upper_bound-1))

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)
    task = CharacterRNNClassification(conf)

    # read data
    if running_index == 5:
        test_index = [9]
        val_index = [8]
        complete_index = np.arange(10)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)
    else:
        test_index = [2 * running_index + 1]
        val_index = [2 * running_index]
        complete_index = np.arange(8)
        train_index = filter(lambda x: x not in test_index and x not in val_index, complete_index)

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index]
    test_file_list = file_list[test_index]

    print('train files ', train_file_list)
    print('val files ', val_file_list)
    print('test files ', test_file_list)

    train_pd = read_merged_data(train_file_list)
    val_pd = read_merged_data(val_file_list)
    test_pd = read_merged_data(test_file_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_SMILES_and_label(train_pd,
                                                feature_name='SMILES',
                                                label_name_list=label_name_list,
                                                SMILES_mapping_json_file=SMILES_mapping_json_file)
    X_val, y_val = extract_SMILES_and_label(val_pd,
                                            feature_name='SMILES',
                                            label_name_list=label_name_list,
                                            SMILES_mapping_json_file=SMILES_mapping_json_file)
    X_test, y_test = extract_SMILES_and_label(test_pd,
                                              feature_name='SMILES',
                                              label_name_list=label_name_list,
                                              SMILES_mapping_json_file=SMILES_mapping_json_file)

    X_train = sequence.pad_sequences(X_train, maxlen=task.padding_length)
    X_val = sequence.pad_sequences(X_val, maxlen=task.padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=task.padding_length)

    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)
    return


def run_ensemble():
    from ensemble import construct_data, Ensemble

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    X_train, y_train = construct_data(conf, file_list)
    print('Consructed data: {}, {}'.format(X_train.shape, y_train.shape))

    task = Ensemble(conf)
    task.train_and_predict(X_train, y_train, weight_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', dest="config_json_file",
                        action="store", required=True)
    parser.add_argument('--weight_file', dest="weight_file",
                        action="store", required=True)
    parser.add_argument('--process_num', dest='process_num', type=int,
                        action='store', required=True)
    parser.add_argument('--SMILES_mapping_json_file', dest='SMILES_mapping_json_file',
                        action='store', required=False, default='../config/SMILES_mapping_keck_MLPCN.json')
    parser.add_argument('--score_file', dest='score_file',
                        action='store', required=False)
    parser.add_argument('--model', dest='model',
                        action='store', required=True)
    parser.add_argument('--cross_validation_upper_bound', dest='cross_validation_upper_bound', type=int,
                        action='store', required=False, default=5)
    given_args = parser.parse_args()

    config_json_file = given_args.config_json_file
    weight_file = given_args.weight_file
    cross_validation_upper_bound = given_args.cross_validation_upper_bound

    process_num = int(given_args.process_num)
    model = given_args.model

    if model == 'single_deep_classification':
        run_single_deep_classification(process_num)
    elif model == 'single_deep_regression':
        run_single_deep_regression(process_num)
    elif model == 'multi_deep_classification':
        score_file = given_args.score_file
        run_multiple_classification(process_num)
    elif model == 'random_forest_classification':
        run_random_forest_classification(process_num)
    elif model == 'random_forest_regression':
        run_random_forest_regression(process_num)
    elif model == 'xgboost_classification':
        run_xgboost_classification(process_num)
    elif model == 'xgboost_regression':
        run_xgboost_regression(process_num)
    elif model == 'ensemble':
        run_ensemble()
    elif model == 'character_rnn_classification':
        SMILES_mapping_json_file = given_args.SMILES_mapping_json_file
        run_character_rnn_classification(process_num)
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
