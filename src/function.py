import pandas as pd
import numpy as np
import random
import os
import json
import csv
from sklearn.cross_validation import StratifiedKFold, KFold


def analysis(data):
    class Node:
        def __init__(self, retest, fp, rmi):
            self.retest = retest
            self.fp = fp
            self.rmi = rmi
            if not np.isnan(self.rmi):
                self.rmi = int(self.rmi)
            else:
                self.rmi = np.NaN
            
        def __str__(self):
            ret = 'retest: {}, fp: {}, rmi: {}'.format(self.retest, self.fp, self.rmi)
            return ret
        
        def __eq__(self, other):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
        
        def __hash__(self):
            return hash(self.retest) ^ hash(self.fp) ^ hash(self.rmi)
        
        def __cmp__(self):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
    
    dict_ = {}
    for ix, row in data.iterrows():
        node = Node(row['Keck_Pria_AS_Retest'], row['Keck_Pria_FP_data'], row['Keck_RMI_cdd'])
        if node not in dict_.keys():
            dict_[node] = 1
        else:
            dict_[node] += 1
    
    for k in dict_.keys():
        print(k, '\t---', dict_[k])
    
    return


def greedy_multi_splitting(data, k, directory, file_list):
    class Node:
        def __init__(self, retest, fp, rmi):
            self.retest = retest
            self.fp = fp
            self.rmi = rmi
            if not np.isnan(self.rmi):
                self.rmi = int(self.rmi)
            else:
                self.rmi = np.NaN
            
        def __str__(self):
            ret = 'retest: {}, fp: {}, rmi: {}'.format(self.retest, self.fp, self.rmi)
            return ret
        
        def __eq__(self, other):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
        
        def __hash__(self):
            return hash(self.retest) ^ hash(self.fp) ^ hash(self.rmi)
        
        def __cmp__(self):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
    
    dict_ = {}
    for ix, row in data.iterrows():
        node = Node(row['Keck_Pria_AS_Retest'], row['Keck_Pria_FP_data'], row['Keck_RMI_cdd'])
        if node not in dict_.keys():
            dict_[node] = []
        dict_[node].append(ix)
        
    list_ = []
    for key in dict_.keys():
        one_group_list = np.array(dict_[key])
        current = []

        if len(one_group_list) < k:
            n = len(one_group_list)
            for i in range(n):
                current.append(np.array(one_group_list[i]))
            for i in range(n, k):
                current.append(np.array([]))
        else:
            kf = KFold(len(one_group_list), k, shuffle=True)
            for _, test_index in kf:
                current.append(one_group_list[test_index])
        random.shuffle(current)
        list_.append(current)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    print(len(list_))

    for split in range(k):
        index_block = np.hstack((list_[0][split],
                                 list_[1][split],
                                 list_[2][split],
                                 list_[3][split],
                                 list_[4][split],
                                 list_[5][split],
                                 list_[6][split],
                                 list_[7][split],
                                 list_[8][split]))
        index_block = index_block.astype(np.int)
        df_block = data.iloc[index_block]
        print(df_block.shape)

        file_path = directory + file_list[split]
        df_block.to_csv(file_path, index=None)
    
    return


def split_data(input_file, output_file_list, k):
    data_pd = pd.read_csv(input_file)
    y_data = data_pd['true_label']
    y_data = y_data.astype(np.float64)
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    cnt = 0
    split = StratifiedKFold(y_data[:, -1], n_folds=k, shuffle=True, random_state=0)
    for train_index, test_index in split:
        # For testing
        # Can list all existing active ones
        # data_batch[data_batch['true_label']>0]['molecule ID(RegID)']
        data_batch = data_pd.iloc[test_index]
        data_batch.to_csv(output_file_list[cnt], index_label=None, compression='gzip')
        cnt += 1
    return


def read_merged_data(input_file_list, usecols=None):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        data_pd = pd.read_csv(input_file, usecols=usecols)
        whole_pd = whole_pd.append(data_pd)
    return whole_pd


def merge_pd(data_pd_list, index_list):
    whole_pd = pd.DataFrame()
    for index in index_list:
        data_pd = data_pd_list[index]
        whole_pd = whole_pd.append(data_pd)
    return whole_pd


def extract_feature_and_label(data_pd,
                              feature_name,
                              label_name_list):
    # By default, feature should be fingerprints
    X_data = data_pd[feature_name].tolist()
    X_data = map(lambda x: list(x), X_data)
    X_data = np.array(X_data)

    y_data = data_pd[label_name_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    return X_data, y_data


def extract_new_fps_and_label(data_pd,
                              feature_name,
                              label_name_list):
    with open('../datasets/keck_pria_lc/new_fingerprint_mapping.csv') as f:
        fingerprint_coding = [{int(k): float(v) for k, v in row.items()}
                              for row in csv.DictReader(f, skipinitialspace=True)]

    X_data = data_pd[feature_name].tolist()
    X_data = map(lambda x: list(x), X_data)
    X_data = np.array(X_data)

    y_data = data_pd[label_name_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    X_data = [map(lambda x: fingerprint_coding[col][x], X_data[:, col]) for col in range(1024)]
    X_data = np.array(X_data)
    X_data = X_data.T

    return X_data, y_data


def extract_SMILES_and_label(data_pd,
                             feature_name,
                             label_name_list,
                             SMILES_mapping_json_file):
    y_data = np.zeros(shape=(data_pd.shape[0], len(label_name_list)))
    X_data = []
    with open(SMILES_mapping_json_file, 'r') as f:
        dictionary = json.load(f)
    print('Character set size {}'.format(len(dictionary)))

    for SMILES in data_pd['SMILES'].tolist():
        SMILES = SMILES.strip()
        row = map(lambda c: dictionary[c], SMILES)
        X_data.append(row)
    X_data = np.array(X_data)

    index = 0
    for _, row in data_pd.iterrows():
        labels = row[label_name_list]
        y_data[index] = np.array(labels)
        index += 1
    y_data = y_data.astype(np.float64)

    # In case we just train on one target
    # y would be (n,) vector
    # then we should change it to (n,1) 1D matrix
    # to keep consistency
    print(y_data.shape)
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    return X_data, y_data


def extract_grammar_and_label(file_list,
                              feature_name='one_hot_matrix',
                              label_name_list=['label_name']):
    X_data = []
    y_data = []
    for file in file_list:
        data_ = np.load(file)
        X_data_ = data_[feature_name]

        y_data_ = []
        for label_name in label_name_list:
            y_data_.append(data_[label_name])
        y_data_ = np.concatenate(y_data_, axis=0)

        X_data.append(X_data_)
        y_data.append(y_data_)

    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    print('feature matrix: {}'.format(X_data.shape))
    print('label matrix: {}'.format(y_data.shape))

    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    return X_data, y_data


def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data


def filter_out_missing_values(data_pd, label_list=['Keck_Pria_AS_Retest']):
    filtered_pd = data_pd.dropna(axis=0, how='any', inplace=False, subset=label_list)
    return filtered_pd

if __name__ == '__main__':
    extract_grammar_and_label(['../datasets/keck_pria_lc/0_grammar.npz'])
    extract_grammar_and_label(['../datasets/keck_pria_lc/1_grammar.npz'])
    extract_grammar_and_label(['../datasets/keck_pria_lc/2_grammar.npz'])
    extract_grammar_and_label(['../datasets/keck_pria_lc/3_grammar.npz'])
    extract_grammar_and_label(['../datasets/keck_pria_lc/4_grammar.npz'])
    extract_grammar_and_label(['../datasets/keck_pria_lc/keck_lc4_grammar.npz'])
