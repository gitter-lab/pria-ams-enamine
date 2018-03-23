from __future__ import print_function

from evaluation import roc_auc_single, roc_auc_multi, bedroc_auc_multi, bedroc_auc_single, \
    precision_auc_multi, precision_auc_single, enrichment_factor_multi, enrichment_factor_single, number_of_hit_single


def output_classification_result(y_train, y_pred_on_train,
                                 y_val, y_pred_on_val,
                                 y_test, y_pred_on_test, EF_ratio_list):

    print('train precision: {}'.format(precision_auc_single(y_pred_on_train, y_train)))
    print('train roc: {}'.format(roc_auc_single(y_pred_on_train, y_train)))
    print('train bedroc: {}'.format(bedroc_auc_single(y_pred_on_train, y_train)))
    N = len(y_train) / 100
    print('train hit in top {}: {} out of {}'.format(N, number_of_hit_single(y_pred_on_train, y_train, N=N), sum(y_train)[0]))
    print()

    if y_pred_on_val is not None:
        print('val precision: {}'.format(precision_auc_single(y_pred_on_val, y_val)))
        print('val roc: {}'.format(roc_auc_single(y_pred_on_val, y_val)))
        print('val bedroc: {}'.format(bedroc_auc_single(y_pred_on_val, y_val)))
        N = len(y_val) / 100
        print('val hit in top {}: {} out of {}'.format(N, number_of_hit_single(y_pred_on_val, y_val, N=N), sum(y_val)[0]))
        print()

    if y_pred_on_test is not None:
        print('test precision: {}'.format(precision_auc_single(y_pred_on_test, y_test)))
        print('test roc: {}'.format(roc_auc_single(y_pred_on_test, y_test)))
        print('test bedroc: {}'.format(bedroc_auc_single(y_pred_on_test, y_test)))
        N = len(y_test) / 100
        print('test hit in top {}: {} out of {}'.format(N, number_of_hit_single(y_pred_on_test, y_test, N=N), sum(y_test)[0]))
        print()
        for EF_ratio in EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_pred_on_test, y_test, EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))
        print()

    return


def output_regression_result(y_train_binary, y_pred_on_train,
                             y_val_binary, y_pred_on_val,
                             y_test_binary, y_pred_on_test, EF_ratio_list):

    print('train precision: {}'.format(precision_auc_single(y_pred_on_train, y_train_binary)))
    print('train roc: {}'.format(roc_auc_single(y_pred_on_train, y_train_binary)))
    print('train bedroc: {}'.format(bedroc_auc_single(y_pred_on_train, y_train_binary)))
    print('train hit: {} out of {}'.format(number_of_hit_single(y_pred_on_train, y_train_binary, N=250),
                                           sum(y_train_binary)[0]))
    print()

    if y_pred_on_val is not None:
        print('val precision: {}'.format(precision_auc_single(y_pred_on_val, y_val_binary)))
        print('val roc: {}'.format(roc_auc_single(y_pred_on_val, y_val_binary)))
        print('val bedroc: {}'.format(bedroc_auc_single(y_pred_on_val, y_val_binary)))
        print('val hit: {} out of {}'.format(number_of_hit_single(y_pred_on_val, y_val_binary, N=250),
                                               sum(y_train_binary)[0]))
        print()

    if y_pred_on_test is not None:
        print('test precision: {}'.format(precision_auc_single(y_pred_on_test, y_test_binary)))
        print('test roc: {}'.format(roc_auc_single(y_pred_on_test, y_test_binary)))
        print('test bedroc: {}'.format(bedroc_auc_single(y_pred_on_test, y_test_binary)))
        print('test hit: {} out of {}'.format(number_of_hit_single(y_pred_on_test, y_test_binary, N=250),
                                              sum(y_test_binary)[0]))
        print()
        for EF_ratio in EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_pred_on_test, y_test_binary, EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))
        print()

    return