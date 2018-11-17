from __future__ import print_function

import os
import json
import numpy as np


def extract(file_path):
    if not os.path.isfile(file_path):
        return -1, -1, -1

    with open(file_path, 'r') as f:
        lines = f.readlines()

    test_roc, test_precision, test_EF = -1, -1, -1
    for line in lines:
        if 'test precision' in line:
            line = line.strip().split(':')
            test_precision = float(line[1])
        if 'test roc' in line:
            line = line.strip().split(':')
            test_roc = float(line[1])
        if 'ratio: 0.01, EF:' in line:
            line = line.strip().replace('EF:', '').split(',')
            test_EF = float(line[1])
    return test_roc, test_precision, test_EF


if __name__ == '__main__':
    model_list = [
        'random_forest_classification',
        'xgboost_classification', 'xgboost_regression',
        'single_deep_classification', 'single_deep_regression'
    ]

    model2number = {
        'random_forest_classification': 216,
        'xgboost_classification': 1000,
        'xgboost_regression': 1000,
        'single_deep_classification': 432,
        'single_deep_regression': 432
    }

    top_k = 20

    for model in model_list:
        print('Model: {}'.format(model))
        number = model2number[model]

        hyper_parameter_result_roc = np.zeros(number)
        hyper_parameter_result_precision = np.zeros(number)
        hyper_parameter_result_EF = np.zeros(number)

        missing_index = set()
        for running_process in range(number):
            test_roc_list, test_precision_list, test_EF_list = [], [], []
            for idx in range(4):
                file_path = '{}/{}_{}_{}.out'.format(model, model, running_process, idx)
                test_roc, test_precision, test_EF = extract(file_path)
                if test_roc == -1 and test_precision == -1:
                    missing_index.add(running_process)
                    # print('file {} missing'.format(file_path))
                if test_roc != -1:
                    test_roc_list.append(test_roc)
                if test_precision != -1:
                    test_precision_list.append(test_precision)
                if test_EF != -1:
                    test_EF_list.append(test_EF)

            if len(test_roc_list) > 0:
                hyper_parameter_result_roc[running_process] = np.mean(test_roc_list)
            if len(test_precision_list) > 0:
                hyper_parameter_result_precision[running_process] = np.mean(test_precision_list)
            if len(test_EF_list) > 0:
                hyper_parameter_result_EF[running_process] = np.mean(test_EF_list)

        # print('top {} AUC[PR]'.format(top_k))
        # hyper_parameter_result_precision_ordered_index = np.argsort(-1 * hyper_parameter_result_precision)
        # for i, running_process in enumerate(hyper_parameter_result_precision_ordered_index):
        #     if i <= top_k:
        #         print('{}-th largest\trunning process: {}\tAUC[PR]: {}'.
        #               format(i, running_process, hyper_parameter_result_precision[running_process]))
        # print()
        # print('top {} AUC[ROC]'.format(top_k))
        # hyper_parameter_result_roc_ordered_index = np.argsort(-1 * hyper_parameter_result_roc)
        # for i, running_process in enumerate(hyper_parameter_result_roc_ordered_index):
        #     if i <= top_k:
        #         print('{}-th largest\trunning process: {}\tAUC[ROC]: {}'.
        #               format(i, running_process, hyper_parameter_result_roc[running_process]))
        # print()
        print('top {} EF'.format(top_k))
        filtered_index = []
        hyper_parameter_result_EF_ordered_index = np.argsort(-1 * hyper_parameter_result_EF)
        for i, running_process in enumerate(hyper_parameter_result_EF_ordered_index):
            if i <= top_k:
                print('{}-th largest\trunning process: {}\tEF: {}'.
                      format(i, running_process, hyper_parameter_result_EF[running_process]))
                # if 'deep'  in model:
                #     config_json_file = '../../config/cross_validation_keck/{}/{}.json'.format(model, running_process)
                #     with open(config_json_file, 'r') as f:
                #         conf = json.load(f)
                #         print('index: {}\t\t# layers: {}'.format(running_process, len(conf['layers']) - 1))
                filtered_index.append(running_process)
        print()

        print('process_list=(', end='')
        for running_process in filtered_index:
            print(' ', running_process, end='')
        print(')')
        print('missing\t', missing_index)
        # missing_index_with_layer_4 = []
        # for index in missing_index:
        #     config_json_file = '../../config/cross_validation_keck/{}/{}.json'.format(model, index)
        #     with open(config_json_file, 'r') as f:
        #         conf = json.load(f)
        #     if 'random_forest' in model:
        #         print('index: {},\t# trees: {}'.format(index, conf['n_estimators']))
        #     elif 'deep' in model:
        #         print('index: {},\t# layers: {},\tNN structure: {}'.format(index, len(conf['layers'])-1, conf['layers']))
        #         if len(conf['layers']) == 5:
        #             missing_index_with_layer_4.append(index)
        # print()
        #
        # if len(missing_index_with_layer_4) > 0:
        #     print('{} missing indices with 4 layers'.format(len(missing_index_with_layer_4)))
        #     print('missing index with 4 layers: {}'.format(missing_index_with_layer_4))
        #     np.random.shuffle(missing_index_with_layer_4)
        #     print('missing index with 4 layers: {}'.format(missing_index_with_layer_4))
        #     print('fetch first 5 missing index: {}'.format(missing_index_with_layer_4[:10]))

        print()
        print()
        print()
        print()
        print()
