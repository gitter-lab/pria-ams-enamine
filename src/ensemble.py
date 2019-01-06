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


def construct_training_data(conf, file_list):
    X_train = []
    K = len(file_list)
    label_name_list = conf['label_name_list']

    for index,test_file in enumerate(file_list):
        running_index = index / 2
        print(test_file)
        test_file_list = [test_file]
        test_pd = read_merged_data(test_file_list)
        feature, _ = extract_feature_and_label(test_pd,
                                               feature_name='1024 MorganFP Radius 2',
                                               label_name_list=label_name_list)
        X_train_current_round = []
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
                print y_pred.shape
                X_train_current_round.append(y_pred)

        X_train_current_round = np.concatenate(X_train_current_round, axis=1)
        X_train.append(X_train_current_round)
        print('Current round {} shape {}'.format(index, X_train_current_round.shape))
        print

    X_train = np.concatenate(X_train)

    train_pd = read_merged_data(file_list)
    _, y_train = extract_feature_and_label(train_pd,
                                           feature_name='1024 MorganFP Radius 2',
                                           label_name_list=label_name_list)
    return X_train, y_train


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
            print y_pred.shape
            X_test_temp.append(y_pred)

    X_test = np.concatenate(X_test_temp, axis=1)

    return X_test, y_test


class Ensemble:
    def __init__(self, conf):
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.random_seed = conf['random_seed']
        np.random.seed(seed=self.random_seed)
        return

    def setup_model(self):
        model = XGBClassifier(max_depth=6,
                              learning_rate=1e-1,
                              n_estimators=100,
                              objective='binary:logistic',
                              booster='gblinear',
                              reg_lambda=0.1,
                              random_state=self.random_seed,
                              silent=False,
                              n_jobs=8)
        # model = LogisticRegression(C=10,
        #                            class_weight='balanced',
        #                            random_state=self.random_seed,
        #                            n_jobs=8)
        return model

    def train_and_predict(self, X_train, y_train, X_test, y_test, weight_file):
        model = self.setup_model()
        model.fit(X_train, y_train, verbose=True)
        # model.fit(X_train, y_train)

        y_pred_on_train = reshape_data_into_2_dim(model.predict_proba(X_train)[:, 1])
        if X_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict_proba(X_test)[:, 1])

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test,
                                     EF_ratio_list=self.EF_ratio_list)

        self.save_model(model, weight_file)

        return

    def save_model(self, model, weight_file):
        from sklearn.externals import joblib
        joblib.dump(model, weight_file, compress=3)
        return

    def load_model(self, weight_file):
        from sklearn.externals import joblib
        model = joblib.load(weight_file)
        return model


def demo_ensemble():
    conf = {
        'models': {
            'random_forest_classification': {
                'task_module': 'RandomForestClassification',
                'config_json_file': '../config/random_forest_classification/{}.json',
                'model_weight': '../model_weight/random_forest_classification/random_forest_classification_{}_{}.pkl',
                'process_num_list': [139, 69, 111, 212, 210, 148, 28, 61, 124, 130, 131, 141, 14, 38, 165, 65, 123, 94, 3, 88, 72],
                'top_process_num': 1
            },
            'xgboost_classification': {
                'task_module': 'XGBoostClassification',
                'config_json_file': '../config/xgboost_classification/{}.json',
                'model_weight': '../model_weight/xgboost_classification/xgboost_classification_{}_{}.pkl',
                'process_num_list': [140, 967, 960, 807, 263, 694, 440, 47, 116, 792, 663, 32, 564, 950, 735, 84, 364, 605, 431, 55, 388],
                'top_process_num': 2
            },
            'xgboost_regression': {
                'task_module': 'XGBoostRegression',
                'config_json_file': '../config/xgboost_regression/{}.json',
                'model_weight': '../model_weight/xgboost_regression/xgboost_regression_{}_{}.pkl',
                'process_num_list': [187, 6, 514, 507, 880, 440, 605, 718, 754, 409, 586, 214, 753, 65, 294, 911, 721, 81, 321, 545, 280],
                'top_process_num': 2
            },
            'single_deep_classification': {
                'task_module': 'SingleClassification',
                'config_json_file': '../config/single_deep_classification/{}.json',
                'model_weight': '../model_weight/single_deep_classification/single_deep_classification_{}_{}.pkl',
                'process_num_list': [328, 423, 325, 53, 339, 42, 407, 253, 28, 416, 208, 124, 366, 273, 132, 106, 259, 214, 27, 24],
                'top_process_num': 2
            },
            'single_deep_regression': {
                'task_module': 'SingleRegression',
                'config_json_file': '../config/single_deep_regression/{}.json',
                'model_weight': '../model_weight/single_deep_regression/single_deep_regression_{}_{}.pkl',
                'process_num_list': [124, 208, 328, 360, 54, 75, 90, 28, 214, 325, 335, 345, 363, 384, 31, 32, 85, 327, 253, 285],
                'top_process_num': 2
            }
        },
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'random_seed': 1337,
        'label_name_list': ['PriA-SSB AS Activity'] # 'PriA-SSB AS % inhibition (Primary Median)'
    }

    # specify dataset
    K = 8
    directory = '../datasets/keck_pria/fold_{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    training_file_list = np.array(file_list)
    X_train, y_train = construct_training_data(conf, training_file_list)
    print('Consructed data: {}, {}'.format(X_train.shape, y_train.shape))

    test_file_list = ['../datasets/keck_pria/fold_8.csv', '../datasets/keck_pria/fold_9.csv']
    X_test, y_test = construct_test_data(conf, test_file_list)
    print'Test data {}, {}'.format(X_test.shape, y_test.shape)

    secondary_layer_model = Ensemble(conf=conf)
    secondary_layer_model.train_and_predict(X_train, y_train, X_test, y_test, weight_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file

    demo_ensemble()
