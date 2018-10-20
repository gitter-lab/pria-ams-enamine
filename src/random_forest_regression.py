from __future__ import print_function

import argparse
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from function import read_merged_data, extract_feature_and_label, reshape_data_into_2_dim
from util import output_regression_result


def get_sample_weight(task, y_data):
    if task.weight_schema == 'no_weight':
        sw = [1.0 for t in y_data]
    elif task.weight_schema == 'weighted_sample':
        values = set(map(lambda x: int(x), y_data))
        values = dict.fromkeys(values, 0)

        data = sorted(y_data)
        for k,g in groupby(data, key=lambda x: int(x[0])):
            temp_group = [t[0] for t in g]
            values[k] = len(temp_group)
        sum_ = reduce(lambda x, y: x + y, values.values())
        sw = map(lambda x: 1.0 * sum_ / values[int(x[0])], y_data)
    else:
        raise ValueError('Weight schema not included. Should be among [{}, {}].'.
                         format('no_weight', 'weighted_sample'))
    sw = np.array(sw)
    # Only accept 1D sample weights
    # sw = reshape_data_into_2_dim(sw)
    return sw


class RandomForestRegression:
    def __init__(self, conf):
        self.conf = conf
        self.max_features = conf['max_features']
        self.n_estimators = conf['n_estimators']
        self.min_samples_leaf = conf['min_samples_leaf']
        self.weight_schema = conf['sample_weight']
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.random_seed = conf['random_seed']

        if 'hit_ratio' in self.conf.keys():
            self.hit_ratio = conf['hit_ratio']
        else:
            self.hit_ratio = 0.01
        np.random.seed(seed=self.random_seed)
        return
    
    def setup_model(self):
        model = RandomForestRegressor(n_estimators=self.n_estimators,
                                      max_features=self.max_features,
                                      min_samples_leaf=self.min_samples_leaf,
                                      n_jobs=8,
                                      random_state=self.random_seed,
                                      oob_score=False,
                                      verbose=1)
        return model

    def train_and_predict(self,
                          X_train, y_train_continuous, y_train_binary,
                          X_test, y_test_continuous, y_test_binary,
                          weight_file):
        model = self.setup_model()
        sw = get_sample_weight(self, y_train_continuous)
        print('Sample Weight\t', sw)

        model.fit(X_train, y_train_continuous)

        y_pred_on_train = reshape_data_into_2_dim(model.predict(X_train))
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict(X_test))
        else:
            y_pred_on_test = None

        output_regression_result(y_train_binary=y_train_binary, y_pred_on_train=y_pred_on_train,
                                 y_val_binary=None, y_pred_on_val=None,
                                 y_test_binary=y_test_binary, y_pred_on_test=y_pred_on_test,
                                 EF_ratio_list=self.EF_ratio_list, hit_ratio=self.hit_ratio)

        self.save_model(model, weight_file)

        return

    def predict_with_existing(self, X_data, weight_file):
        model = self.load_model(weight_file)
        y_pred = reshape_data_into_2_dim(model.predict(X_data))
        return y_pred

    def eval_with_existing(self,
                           X_train, y_train_continuous, y_train_binary,
                           X_test, y_test_continuous, y_test_binary,
                           weight_file):
        model = self.load_model(weight_file)

        y_pred_on_train = reshape_data_into_2_dim(model.predict(X_train))
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict(X_test))
        else:
            y_pred_on_test = None

        output_regression_result(y_train_binary=y_train_binary, y_pred_on_train=y_pred_on_train,
                                 y_val_binary=None, y_pred_on_val=None,
                                 y_test_binary=y_test_binary, y_pred_on_test=y_pred_on_test,
                                 EF_ratio_list=self.EF_ratio_list, hit_ratio=self.hit_ratio)
        return

    def save_model(self, model, weight_file):
        from sklearn.externals import joblib
        joblib.dump(model, weight_file, compress=3)
        return

    def load_model(self, weight_file):
        from sklearn.externals import joblib
        model = joblib.load(weight_file)
        return model


def demo_random_forest_regression():
    conf = {
        'max_features': 'log2',
        'n_estimators': 4000,
        'min_samples_leaf': 1,
        'sample_weight': 'no_weight',
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'random_seed': 1337,
        'label_name_list': ['Keck_Pria_AS_Retest', 'Keck_Pria_Continuous']
    }

    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    test_index = 0
    complete_index = np.arange(K)
    train_index = np.where(complete_index != test_index)[0]
    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index:test_index + 1]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    train_pd = read_merged_data(train_file_list)
    test_pd = read_merged_data(test_file_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
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
    task.eval_with_existing(X_train, y_train_continuous, y_train_binary,
                            X_test, y_test_continuous, y_test_binary,
                            weight_file)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file

    # specify dataset
    K = 5
    directory = '../datasets/keck_pria_lc/{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list = np.array(file_list)

    demo_random_forest_regression()
