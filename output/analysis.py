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

    model_process_num_list = {
        'random_forest_classification': [139, 69, 111, 212, 210, 148, 28, 61, 124, 130, 131, 141, 14, 38, 165, 65, 123, 94, 3, 88, 72],
        'xgboost_classification': [140, 967, 960, 807, 263, 694, 440, 47, 116, 792, 663, 32, 564, 950, 735, 84, 364, 605, 431, 55, 388],
        'xgboost_regression': [187, 6, 514, 507, 880, 440, 605, 718, 754, 409, 586, 214, 753, 65, 294, 911, 721, 81, 321, 545, 280],
        'single_deep_classification': [328, 423, 325, 53, 339, 42, 407, 253, 28, 416, 208, 124, 366, 273, 132, 106, 259, 214, 27, 24],
        'single_deep_regression': [124, 208, 328, 360, 54, 75, 90, 28, 214, 325, 335, 345, 363, 384, 31, 32, 85, 327, 253, 285],
    }

    for model in model_list:
        print('Model: {}'.format(model))
        number = len(model_process_num_list[model])

        hyper_parameter_result_roc = []
        hyper_parameter_result_precision = []
        hyper_parameter_result_EF = []

        for running_process in model_process_num_list[model]:
            test_roc_list, test_precision_list, test_EF_list = [], [], []
            for idx in range(4):
                file_path = '{}/{}_{}_{}.out'.format(model, model, running_process, idx)
                test_roc, test_precision, test_EF = extract(file_path)
                if test_roc == -1 and test_precision == -1:
                    missing_index.add(running_process)
                if test_roc != -1:
                    test_roc_list.append(test_roc)
                if test_precision != -1:
                    test_precision_list.append(test_precision)
                if test_EF != -1:
                    test_EF_list.append(test_EF)

            hyper_parameter_result_roc.append(np.mean(test_roc_list))
            hyper_parameter_result_precision.append(np.mean(test_precision_list))
            hyper_parameter_result_EF.append(np.mean(test_EF_list))

        for running_process, roc, pr, EF in zip(model_process_num_list[model], hyper_parameter_result_roc, hyper_parameter_result_precision, hyper_parameter_result_EF):
            print('{}\t{}\t{}\t{}'.format(running_process, roc, pr, EF))
        print()
