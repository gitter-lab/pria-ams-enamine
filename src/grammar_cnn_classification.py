import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution1D
from function import read_merged_data, extract_grammar_and_label
from evaluation import roc_auc_single, precision_auc_single, enrichment_factor_single
from CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision
from util import output_classification_result


class GrammarCNNClassification:
    def __init__(self, conf):
        self.conf = conf

        self.padding_length = conf['cnn']['padding_length']
        self.vocabulary_size = conf['cnn']['grammar_vocabulary_length']
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
        return

    def setup_model(self):
        x = Input(shape=(self.padding_length, self.vocabulary_size))
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)
        h = Dense(self.output_layer_dimension, activation='relu', name='dense_2')(h)
        model = Model(x, h)
        print(model.summary())
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

        model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer)
        model.fit(X_train, y_train, nb_epoch=self.fit_nb_epoch, batch_size=self.fit_batch_size,
                  callbacks=callbacks,
                  verbose=self.fit_verbose)

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

    def eval_with_existing(self,
                           X_train, y_train,
                           X_val, y_val,
                           X_test, y_test,
                           weight_file):
        model = self.setup_model()
        model.load_weights(weight_file)


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


def demo_grammar_cnn_classification():
    conf = {
        'cnn': {
            'padding_length': 325,
            'grammar_vocabulary_length': 79,
        },
        'compile': {
            'loss': 'binary_crossentropy',
            'optimizer': {
                'option': 'adam',
                'adam': {
                    'lr': 0.0001,
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
                'option': 'auc',
                'patience': 50
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
        'class_weight_option': 'no_weight'
    }
    task = GrammarCNNClassification(conf)

    train_file_list = file_list[: K-1]
    val_file_list = file_list[K-1: K]
    test_file_list = file_list[K: K+1]
    print('train file list: {}\nval file list: {}\ntest file list: {}'.format(
        train_file_list, val_file_list, test_file_list))

    # extract data, and split training data into training and val
    X_train, y_train = extract_grammar_and_label(train_file_list)
    X_val, y_val = extract_grammar_and_label(val_file_list)
    X_test, y_test = extract_grammar_and_label(test_file_list)
    print('done data preparation')

    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file

    # specify dataset
    K = 5
    directory = '../datasets/keck_pria_lc/{}_grammar.npz'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list.append('../datasets/keck_pria_lc/keck_lc4_grammar.npz')
    file_list = np.array(file_list)

    demo_grammar_cnn_classification()
