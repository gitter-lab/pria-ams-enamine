from __future__ import print_function

import json
import random


def generate_ensemble_hyperparameter(index, model_process_num_list):
    conf = {
        'models': {
            'random_forest_classification': {
                'task_module': 'RandomForestClassification',
                'config_json_file': '../config/random_forest_classification/{}.json',
                'model_weight': '../model_weight/random_forest_classification/random_forest_classification_{}_{}.pkl',
                'process_num_list': [139, 69, 111, 212, 210, 148, 28, 61, 124, 130, 131, 141, 14, 38, 165, 65, 123, 94,
                                     3, 88, 72],
                'top_process_num': 0
            },
            'xgboost_classification': {
                'task_module': 'XGBoostClassification',
                'config_json_file': '../config/xgboost_classification/{}.json',
                'model_weight': '../model_weight/xgboost_classification/xgboost_classification_{}_{}.pkl',
                'process_num_list': [140, 967, 960, 807, 263, 694, 440, 47, 116, 792, 663, 32, 564, 950, 735, 84, 364,
                                     605, 431, 55, 388],
                'top_process_num': 0
            },
            'xgboost_regression': {
                'task_module': 'XGBoostRegression',
                'config_json_file': '../config/xgboost_regression/{}.json',
                'model_weight': '../model_weight/xgboost_regression/xgboost_regression_{}_{}.pkl',
                'process_num_list': [187, 6, 514, 507, 880, 440, 605, 718, 754, 409, 586, 214, 753, 65, 294, 911, 721,
                                     81, 321, 545, 280],
                'top_process_num': 0
            },
            'single_deep_classification': {
                'task_module': 'SingleClassification',
                'config_json_file': '../config/single_deep_classification/{}.json',
                'model_weight': '../model_weight/single_deep_classification/single_deep_classification_{}_{}.pkl',
                'process_num_list': [328, 423, 325, 53, 339, 42, 407, 253, 28, 416, 208, 124, 366, 273, 132, 106, 259,
                                     214, 27, 24],
                'top_process_num': 0
            },
            'single_deep_regression': {
                'task_module': 'SingleRegression',
                'config_json_file': '../config/single_deep_regression/{}.json',
                'model_weight': '../model_weight/single_deep_regression/single_deep_regression_{}_{}.pkl',
                'process_num_list': [124, 208, 328, 360, 54, 75, 90, 28, 214, 325, 335, 345, 363, 384, 31, 32, 85, 327,
                                     253, 285],
                'top_process_num': 0
            }
        },
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'random_seed': 1337,
        'label_name_list': ['PriA-SSB AS Activity']  # 'PriA-SSB AS % inhibition (Primary Median)'
    }

    for process_num, model in zip(model_process_num_list, model_list):
        conf['models'][model]['top_process_num'] = process_num

    with open('ensemble/{}.json'.format(index), 'w') as file_:
        json.dump(conf, file_)

    return


if __name__ == '__main__':
    model_list = [
        'random_forest_classification',
        'xgboost_classification',
        'xgboost_regression',
        'single_deep_classification',
        'single_deep_regression'
    ]

    index = 0

    model_process_num_list = [1, 0, 0, 0, 0]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [0, 1, 0, 0, 0]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [0, 0, 1, 0, 0]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [0, 0, 0, 1, 0]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [0, 0, 0, 0, 1]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [1, 1, 1, 1, 1]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [2, 2, 2, 2, 2]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [2, 2, 1, 2, 1]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [5, 5, 3, 5, 3]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [10, 10, 5, 10, 5]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [5, 5, 10, 5, 10]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [20, 20, 10, 20, 10]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1

    model_process_num_list = [10, 10, 20, 10, 20]
    generate_ensemble_hyperparameter(index, model_process_num_list)
    index += 1
