from __future__ import print_function

import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from function import read_merged_data, extract_feature_and_label, reshape_data_into_2_dim
from CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision
from util import output_classification_result


def count_occurance(key_list, target_list):
    weight = {i: 0 for i in key_list}
    for x in target_list:
        weight[x] += 1
    return weight


def get_class_weight(task, y_data):
    if task.weight_schema == 'no_weight':
        cw = []
        for i in range(task.output_layer_dimension):
            cw.append({0: 0.5, 1: 0.5})
    elif task.weight_schema == 'weighted_sample':
        cw = []
        for i in range(task.output_layer_dimension):
            w = count_occurance([0, 1], y_data[:, i])
            zero_weight = 1.0
            one_weight = 1.0 * w[0] / w[1]
            cw.append({0: zero_weight, 1: one_weight})
    else:
        raise ValueError('Weight schema not included. Should be among [{}, {}].'.
                         format('no_weight', 'weighted_sample'))

    return cw


class SingleClassification:
    def __init__(self, conf):
        self.conf = conf
        if 'input_layer_dimension' in conf.keys():
            self.input_layer_dimension = conf['input_layer_dimension']
        else:
            self.input_layer_dimension = 1024
        self.output_layer_dimension = 1

        self.early_stopping_patience = conf['fitting']['early_stopping']['patience']
        self.early_stopping_option = conf['fitting']['early_stopping']['option']

        self.fit_nb_epoch = conf['fitting']['nb_epoch']
        self.fit_batch_size = conf['fitting']['batch_size']
        self.fit_verbose = conf['fitting']['verbose']

        self.compile_loss = conf['compile']['loss']
        self.compile_optimizer_option = conf['compile']['optimizer']['option']
        if self.compile_optimizer_option == 'sgd':
            sgd_lr = conf['compile']['optimizer']['sgd']['lr']
            sgd_momentum = conf['compile']['optimizer']['sgd']['momentum']
            sgd_decay = conf['compile']['optimizer']['sgd']['decay']
            sgd_nestrov = conf['compile']['optimizer']['sgd']['nestrov']
            self.compile_optimizer = SGD(lr=sgd_lr, momentum=sgd_momentum, decay=sgd_decay, nesterov=sgd_nestrov)
        else:
            adam_lr = conf['compile']['optimizer']['adam']['lr']
            adam_beta_1 = conf['compile']['optimizer']['adam']['beta_1']
            adam_beta_2 = conf['compile']['optimizer']['adam']['beta_2']
            adam_epsilon = conf['compile']['optimizer']['adam']['epsilon']
            self.compile_optimizer = Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)

        self.batch_is_use = conf['batch']['is_use']
        if self.batch_is_use:
            batch_normalizer_epsilon = conf['batch']['epsilon']
            batch_normalizer_mode = conf['batch']['mode']
            batch_normalizer_axis = conf['batch']['axis']
            batch_normalizer_momentum = conf['batch']['momentum']
            batch_normalizer_weights = conf['batch']['weights']
            batch_normalizer_beta_init = conf['batch']['beta_init']
            batch_normalizer_gamma_init = conf['batch']['gamma_init']
            self.batch_normalizer = BatchNormalization(epsilon=batch_normalizer_epsilon,
                                                       mode=batch_normalizer_mode,
                                                       axis=batch_normalizer_axis,
                                                       momentum=batch_normalizer_momentum,
                                                       weights=batch_normalizer_weights,
                                                       beta_init=batch_normalizer_beta_init,
                                                       gamma_init=batch_normalizer_gamma_init)
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.weight_schema = conf['class_weight_option']

        if 'hit_ratio' in self.conf.keys():
            self.hit_ratio = conf['hit_ratio']
        else:
            self.hit_ratio = 0.01
        return

    def setup_model(self):
        model = Sequential()
        layers = self.conf['layers']
        dropout = self.conf['drop_out']
        layer_number = len(layers)
        for i in range(layer_number):
            init = layers[i]['init']
            activation = layers[i]['activation']
            if i == 0:
                hidden_units = int(layers[i]['hidden_units'])
                model.add(Dense(hidden_units, input_dim=self.input_layer_dimension, init=init, activation=activation))
                model.add(Dropout(dropout))
            elif i == layer_number - 1:
                if self.batch_is_use:
                    model.add(self.batch_normalizer)
                model.add(Dense(self.output_layer_dimension, init=init, activation=activation))
            else:
                hidden_units = int(layers[i]['hidden_units'])
                model.add(Dense(hidden_units, init=init, activation=activation))
                model.add(Dropout(dropout))

        return model

    def train_and_predict(self,
                          X_train, y_train,
                          X_val, y_val,
                          X_test, y_test,
                          weight_file):
        model = self.setup_model()
        if self.early_stopping_option == 'auc':
            early_stopping = KeckCallBackOnROC(X_train, y_train, X_val, y_val,
                                               patience=self.early_stopping_patience,
                                               file_path=weight_file)
            callbacks = [early_stopping]
        elif self.early_stopping_option == 'precision':
            early_stopping = KeckCallBackOnPrecision(X_train, y_train, X_val, y_val,
                                                     patience=self.early_stopping_patience,
                                                     file_path=weight_file)
            callbacks = [early_stopping]
        else:
            callbacks = []

        cw = get_class_weight(self, y_train)
        print('Class Weight ', cw)

        model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer)
        model.fit(X_train, y_train,
                  nb_epoch=self.fit_nb_epoch,
                  batch_size=self.fit_batch_size,
                  verbose=self.fit_verbose,
                  class_weight=cw,
                  shuffle=True,
                  callbacks=callbacks)

        if self.early_stopping_option == 'auc' or self.early_stopping_option == 'precision':
            model = early_stopping.get_best_model()

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        if X_test is not None:
            y_pred_on_test = model.predict(X_test)
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=y_val, y_pred_on_val=y_pred_on_val,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test,
                                     EF_ratio_list=self.EF_ratio_list, hit_ratio=self.hit_ratio)
        return

    def predict_with_existing(self, X_data, weight_file):
        model = self.load_model(weight_file)
        y_pred = reshape_data_into_2_dim(model.predict(X_data))
        return y_pred

    def eval_with_existing(self,
                           X_train, y_train,
                           X_val, y_val,
                           X_test, y_test,
                           weight_file):
        model = self.load_model(weight_file)

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        if X_test is not None:
            y_pred_on_test = model.predict(X_test)
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=y_val, y_pred_on_val=y_pred_on_val,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test,
                                     EF_ratio_list=self.EF_ratio_list, hit_ratio=self.hit_ratio)
        return

    def save_model(self, model, weight_file):
        model.save_weights(weight_file)
        return

    def load_model(self, weight_file):
        model = self.setup_model()
        model.load_weights(weight_file)
        return model


def demo_single_classification():
    conf = {
        'layers': [
            {
                'hidden_units': 2000,
                'init': 'glorot_normal',
                'activation': 'relu'
            }, {
                'hidden_units': 2000,
                'init': 'glorot_normal',
                'activation': 'sigmoid'
            }, {
                'init': 'glorot_normal',
                'activation': 'sigmoid'
            }
        ],
        'drop_out': 0.25,
        'compile': {
            'loss': 'binary_crossentropy',
            'optimizer': {
                'option': 'adam',
                'sgd': {
                    'lr': 0.003,
                    'momentum': 0.9,
                    'decay': 0.9,
                    'nestrov': True
                },
                'adam': {
                    'lr': 0.003,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-8
                }
            }
        },
        'fitting': {
            'nb_epoch': 3,
            'batch_size': 2048,
            'verbose': 0,
            'early_stopping': {
                'option': 'precision',
                'patience': 200
            }
        },
        'batch': {
            'is_use': True,
            'epsilon': 2e-5,
            'mode': 0,
            'axis': -1,
            'momentum': 0.9,
            'weights': None,
            'beta_init': 'zero',
            'gamma_init': 'one'
        },
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'class_weight_option': 'weighted_sample',
        'label_name_list': ['PriA-SSB AS Activity']
    }
    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    train_pd = read_merged_data(file_list[0:3])
    train_pd.fillna(0, inplace=True)
    val_pd = read_merged_data(file_list[3:4])
    val_pd.fillna(0, inplace=True)
    test_pd = read_merged_data(file_list[4:5])
    test_pd.fillna(0, inplace=True)

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
    print('done data preparation')

    print(conf['label_name_list'])
    task = SingleClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)
    task.eval_with_existing(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    parser.add_argument('--mode', action='store', dest='mode', required=False, default='single_classification')
    given_args = parser.parse_args()
    weight_file = given_args.weight_file
    mode = given_args.mode

    if mode == 'single_classification':
        # specify dataset
        K = 5
        directory = '../datasets/keck_pria_test/fold_{}.csv'
        file_list = []
        for i in range(K):
            file_list.append(directory.format(i))
        file_list = np.array(file_list)
        demo_single_classification()