from __future__ import print_function

import keras
import sys
import time
from evaluation import *


# define custom classes
# following class is used for keras to compute the AUC each epoch
# and do early stoppping based on that
class KeckCallBackOnROC(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_roc = 0
        self.best_roc = 0
        self.counter = 0
        self.patience = patience
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_roc = roc_auc_single(self.model.predict(self.X_val), self.y_val)
        self.best_roc = self.curr_roc
        self.model.save_weights(self.file_path)
        self.time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        print('Epoch {}/{}'.format(epoch + 1, self.nb_epoch))
        training_end_time = time.time()
        print('Epoch training duration: {}'.format(training_end_time-self.time))
        self.curr_roc = roc_auc_single(self.model.predict(self.X_val), self.y_val)
        if self.curr_roc < self.best_roc:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_roc = self.curr_roc
            self.model.save_weights(self.file_path)

        train_roc = roc_auc_single(self.model.predict(self.X_train), self.y_train)
        train_pr = precision_auc_single(self.model.predict(self.X_train), self.y_train)
        curr_pr = precision_auc_single(self.model.predict(self.X_val), self.y_val)
        print('Train\tAUC[ROC]: {:.6f}\tAUC[PR]: {:.6f}'.format((train_roc, train_pr)))
        print('Val\tAUC[ROC]: {:.6f}\tAUC[PR]: {:.6f}'.format((self.curr_roc, curr_pr)))
        end_time = time.time()
        print('Epoch evaluation duration: {}'.format(end_time-training_end_time))
        print('Epoch duration: {}'.format(end_time-self.time))
        self.time = end_time
        print

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_roc(self):
        return self.best_roc


# define custom classes
# following class is used for keras to compute the precision each epoch
# and do early stoppping based on that
class KeckCallBackOnPrecision(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_pr = 0
        self.best_pr = 0
        self.counter = 0
        self.patience = patience
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_pr = precision_auc_single(self.model.predict(self.X_val), self.y_val)
        self.best_pr = self.curr_pr
        self.model.save_weights(self.file_path)
        self.time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        print('Epoch {}/{}'.format(epoch + 1, self.nb_epoch))
        training_end_time = time.time()
        print('Epoch training duration: {}'.format(training_end_time-self.time))
        self.curr_pr = precision_auc_single(self.model.predict(self.X_val), self.y_val)
        if self.curr_pr < self.best_pr:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_pr = self.curr_pr
            self.model.save_weights(self.file_path)

        train_roc = roc_auc_single(self.model.predict(self.X_train), self.y_train)
        train_pr = precision_auc_single(self.model.predict(self.X_train), self.y_train)
        curr_roc = roc_auc_single(self.model.predict(self.X_val), self.y_val)
        print('Train\tAUC[ROC]: {:.6f}\tAUC[PR]: {:.6f}'.format((train_roc, train_pr)))
        print('Val\tAUC[ROC]: {:.6f}\tAUC[PR]: {:.6f}'.format((curr_roc, self.curr_pr)))
        end_time = time.time()
        print('Epoch evaluation duration: {}'.format(end_time-training_end_time))
        print('Epoch duration: {}'.format(end_time-self.time))
        self.time = end_time
        print()

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_roc(self):
        return self.best_pr
