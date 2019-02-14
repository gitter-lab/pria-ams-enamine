from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
import json
from function import read_merged_data, extract_feature_and_label, reshape_data_into_2_dim
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from random_forest_classification import RandomForestClassification
from xgboost_classification import XGBoostClassification
from xgboost_regression import XGBoostRegression
from deep_classification import SingleClassification
from deep_regression import SingleRegression
from util import output_classification_result


def construct_test_data(conf, file_list):
    label_name_list = conf['label_name_list']
    test_pd = read_merged_data(file_list)
    feature, y_test = extract_feature_and_label(test_pd,
                                                feature_name='1024 MorganFP Radius 2',
                                                label_name_list=label_name_list)

    running_index = 4
    X_test_temp = []
    for model, model_conf in conf['models'].items():
        print('Loading {} ......'.format(model))
        process_num_list = model_conf['process_num_list']
        top_process_num = model_conf['top_process_num']
        print('Pick up top {} out of {}'.format(top_process_num, len(process_num_list)))
        for process_num in process_num_list[:top_process_num]:
            task_module = globals()[model_conf['task_module']]
            config_json_file = model_conf['config_json_file'].format(process_num)
            model_weight_file = model_conf['model_weight'].format(process_num, running_index)
            with open(config_json_file, 'r') as f:
                task_conf = json.load(f)
            task = task_module(conf=task_conf)
            y_pred = task.predict_with_existing(feature, model_weight_file)
            print(y_pred.shape)
            X_test_temp.append(y_pred)

    X_test = np.concatenate(X_test_temp, axis=1)

    return X_test, y_test


class Ensemble:
    def __init__(self, conf):
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.random_seed = conf['random_seed']
        np.random.seed(seed=self.random_seed)
        return

    def predict(self, X):
        y_pred = np.max(X, axis=1)
        return y_pred

    def train_and_predict(self, X_test, y_test, weight_file):

        y_pred_on_test = reshape_data_into_2_dim(self.predict(X_test))

        output_classification_result(y_train=None, y_pred_on_train=None,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test,
                                     EF_ratio_list=self.EF_ratio_list)

        return


def demo_ensemble():
    # specify dataset
    K = 8
    directory = '../datasets/keck_pria/fold_{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))

    test_file_list = ['../datasets/keck_pria/fold_9.csv']
    X_test, y_test = construct_test_data(conf, test_file_list)
    print('Test data {}, {}'.format(X_test.shape, y_test.shape))

    secondary_layer_model = Ensemble(conf=conf)
    secondary_layer_model.train_and_predict(X_test, y_test, weight_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action='store', required=True)
    parser.add_argument('--weight_file', action='store', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file
    config_json_file = given_args.config_json_file

    with open(config_json_file, 'r') as f:
        conf = json.load(f)

    demo_ensemble()
