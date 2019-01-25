import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import RepeatVector, TimeDistributedDense
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from function import read_merged_data, extract_SMILES_and_label
from CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision
from util import output_classification_result


class CharacterRNNClassification:
    def __init__(self, conf):
        self.conf = conf

        # this padding length works the same as time steps
        self.padding_length = conf['lstm']['padding_length']
        self.vocabulary_size = conf['lstm']['different_alphabets_num'] + 1
        self.embedding_size = conf['lstm']['embedding_size']
        self.activation = conf['lstm']['activation']
        self.layer_num = conf['lstm']['layer_num']

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

        if 'hit_ratio' in self.conf.keys():
            self.hit_ratio = conf['hit_ratio']
        else:
            self.hit_ratio = 0.01
        return

    def setup_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocabulary_size,
                            output_dim=self.embedding_size,
                            input_length=self.padding_length))
        layers = self.conf['layers']
        layer_num = self.layer_num

        for i in range(layer_num):
            hidden_size = layers[i]['hidden_size']
            dropout_W = layers[i]['dropout_W']
            dropout_U = layers[i]['dropout_U']
            if i == layer_num - 1:
                model.add(LSTM(hidden_size,
                               dropout_W=dropout_W,
                               dropout_U=dropout_U,
                               return_sequences=False))
            elif i == 0:
                model.add(LSTM(hidden_size,
                               input_shape=(self.padding_length, self.embedding_size),
                               dropout_W=dropout_W,
                               dropout_U=dropout_U,
                               return_sequences=True))
            else:
                model.add(LSTM(hidden_size,
                               dropout_W=dropout_W,
                               dropout_U=dropout_U,
                               return_sequences=True))

        model.add(Dense(self.output_layer_dimension,
                        activation=self.activation))
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


def demo_character_rnn_classification():
    conf = {
        'lstm': {
            'padding_length': 91,
            'different_alphabets_num': 35,
            'embedding_size': 30,
            'activation': 'sigmoid',
            'layer_num': 2
        },
        'layers': [
            {
                'hidden_size': 100,
                'dropout_W': 0.2,
                'dropout_U': 0.2
            },
            {
                'hidden_size': 10,
                'dropout_W': 0.2,
                'dropout_U': 0.2
            }
        ],
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
        'class_weight_option': 'no_weight',
        'label_name_list': ['Keck_Pria_AS_Retest']
    }
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list
    task = CharacterRNNClassification(conf)

    train_pd = read_merged_data(file_list[0:3])
    train_pd.fillna(0, inplace=True)
    val_pd = read_merged_data(file_list[3:4])
    val_pd.fillna(0, inplace=True)
    test_pd = read_merged_data(file_list[4:5])
    test_pd.fillna(0, inplace=True)

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
    print('done data preparation')

    X_train = sequence.pad_sequences(X_train, maxlen=task.padding_length)
    X_val = sequence.pad_sequences(X_val, maxlen=task.padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=task.padding_length)

    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, weight_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    parser.add_argument('--SMILES_mapping_json_file', action='store', dest='SMILES_mapping_json_file',
                        default='../config/SMILES_mapping_keck_LifeChem.json')
    given_args = parser.parse_args()
    weight_file = given_args.weight_file
    SMILES_mapping_json_file = given_args.SMILES_mapping_json_file

    # specify dataset
    K = 5
    directory = '../datasets/keck_pria_lc/{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list = np.array(file_list)

    demo_character_rnn_classification()
