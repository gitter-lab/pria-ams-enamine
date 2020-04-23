from __future__ import print_function

import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import os
import json
import numpy as np



def extract(file_path):
    if not os.path.isfile(file_path):
        return -1, -1, -1

    with open(file_path, 'r') as f:
        lines = f.readlines()

    test_roc, test_precision, test_NEF = -1, -1, -1
    for line in lines:
        if 'test precision' in line:
            line = line.strip().split(':')
            test_precision = float(line[1])
        if 'test roc' in line:
            line = line.strip().split(':')
            test_roc = float(line[1])
        if 'ratio: 0.01, NEF:' in line:
            line = line.strip().replace('NEF:', '').split(',')
            test_NEF = float(line[1])
    return test_roc, test_precision, test_NEF


if __name__ == '__main__':
    model_list = [
        'random_forest_classification',
        'xgboost_classification', 'xgboost_regression',
        'single_deep_classification', 'single_deep_regression'
    ]

    model_process_num_list = {
        'random_forest_classification': [139, 69, 111, 212, 210, 148, 28, 61, 124, 130, 131, 141, 14, 38, 165, 65, 123, 94, 3, 88, 72],
        'xgboost_classification': [140, 967, 960, 807, 263, 694, 440, 47, 116, 792, 663, 32, 564, 950, 735, 84, 364, 605, 431, 55, 388],
        'xgboost_regression': [187, 6, 514, 507, 880, 440, 605, 718, 754, 409, 586, 214, 753, 65, 294, 911, 721, 81, 321, 545, 280],
        'single_deep_classification': [356, 404, 215, 93, 254, 88, 423, 47, 363, 132, 5, 385, 370, 29, 415, 54, 124, 183, 180, 416],
        'single_deep_regression': [199, 323, 114, 123, 47, 175, 17, 178, 106, 265, 67, 157, 369, 115, 191, 20, 27, 108, 270, 45],
        'ensemble': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'ensemble_02': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    # for model in model_list:
    #     print('Model: {}'.format(model))
    #     number = len(model_process_num_list[model])
    #     hyper_parameter_result_roc = []
    #     hyper_parameter_result_precision = []
    #     hyper_parameter_result_NEF = []
    #
    #     for running_process in model_process_num_list[model]:
    #         test_roc_list, test_precision_list, test_NEF_list = [], [], []
    #         for idx in range(4):
    #             file_path = '../output/{}/{}_{}_{}.out'.format(model, model, running_process, idx)
    #             test_roc, test_precision, test_NEF = extract(file_path)
    #             if test_roc == -1 and test_precision == -1:
    #                 print('missing file: {}'.format(file_path))
    #             if test_roc != -1:
    #                 test_roc_list.append(test_roc)
    #             if test_precision != -1:
    #                 test_precision_list.append(test_precision)
    #             if test_NEF != -1:
    #                 test_NEF_list.append(test_NEF)
    #
    #         hyper_parameter_result_roc.append(np.mean(test_roc_list))
    #         hyper_parameter_result_precision.append(np.mean(test_precision_list))
    #         hyper_parameter_result_NEF.append(np.mean(test_NEF_list))
    #
    #     for running_process, roc, pr, NEF in zip(model_process_num_list[model], hyper_parameter_result_roc, hyper_parameter_result_precision, hyper_parameter_result_NEF):
    #         print('{}\t{}\t{}\t{}'.format(running_process, roc, pr, NEF))
    #     print()

    print('On The Last Folder')
    model_list = [
        'random_forest_classification',
        'xgboost_classification', 'xgboost_regression',
        'single_deep_classification', 'single_deep_regression',
        'baseline',
        'ensemble', 'ensemble_02'
    ]

    model_process_num_list = {
        'random_forest_classification': [139],
        'xgboost_classification': [140],
        'xgboost_regression': [187],
        'single_deep_classification': [356],
        'single_deep_regression': [199],
        'baseline': [0],
        'ensemble': [0],
        'ensemble_02': [0],
    }

    def update_name(name):
        if name == 'random_forest_classification':
            name = 'RF-C'
        if name == 'xgboost_classification':
            name = 'XGB-C'
        if name == 'xgboost_regression':
            name = 'XGB-R'
        if name == 'single_deep_classification':
            name = 'NN-C'
        if name == 'single_deep_regression':
            name = 'NN-R'
        if name == 'ensemble':
            name = 'Ensemble, model-based'
        if name =='ensemble_02':
            name = 'Ensemble, max-vote'
        if name == 'baseline':
            name = 'Similarity Baseline'
        return name

    name_list, roc_list, pr_list, NEF_list = [], [], [], []
    for model in model_list:
        print('Model: {}'.format(model))
        number = len(model_process_num_list[model])

        for running_process in model_process_num_list[model]:
            if model == 'ensemble' or model == 'ensemble_02':
                file_path = '../output/{}/{}.out'.format(model, running_process)
            else:
                file_path = '../output/{}/{}_{}_4.out'.format(model, model, running_process)

            test_roc, test_pr, test_NEF = extract(file_path)
            print('{}\t{}'.format(running_process, test_NEF))
            name_list.append(update_name(model))
            roc_list.append(test_roc)
            pr_list.append(test_pr)
            NEF_list.append(test_NEF)
        print()

    for name,roc,pr,NEF in zip(name_list, roc_list, pr_list, NEF_list):
        print('{}&{:.3f}&{:.3f}&{:.3f}\\\\'.format(name, roc, pr, NEF)) 
