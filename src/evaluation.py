import pandas as pd
import csv
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os


def roc_auc_single(predicted, actual):
    try:
        auc_ret = roc_auc_score(actual, predicted)
    except ValueError:
        auc_ret = np.nan

    return auc_ret


def precision_auc_single(predicted, actual):
    try:
        prec_auc = average_precision_score(actual, predicted)
    except ValueError:
        prec_auc = np.nan
    return prec_auc


def number_of_hit_single(predicted, actual, N):
    assert N <= actual.shape[0], \
        'Top Number N=[{}] must be no greater than total compound number [{}]'.format(N, actual.shape[0])

    if predicted.ndim == 2:
        predicted = predicted[:, 0]
    if actual.ndim == 2:
        actual = actual[:, 0]

    top_N_index = predicted.argsort()[::-1][:N]
    top_N = actual[top_N_index]
    n_hit = sum(top_N)
    return n_hit


def ratio_of_hit_single(predicted, actual, R):
    assert 0.0 <= R <= 1.0, 'Top Ratio R=[{}] must be within [0.0, 1.0]'.format(R)
    N = int(R * actual.shape[0])
    return number_of_hit_single(actual, predicted, N)


def enrichment_factor_single(scores_arr, labels_arr, percentile):
    '''
    calculate the enrichment factor based on some upper fraction
    of library ordered by docking scores. upper fraction is determined
    by percentile (actually a fraction of value 0.0-1.0)

    -1 represents missing value
    and remove them when in evaluation
    '''
    non_missing_indices = np.argwhere(labels_arr != -1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]

    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr, axis=0)[::-1][:sample_size]  # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr, axis=0)[::-1][:sample_size]  # get the index positions for these in library
    n_actives = np.nansum(labels_arr)  # count number of positive labels in library
    total_actives = np.nansum(labels_arr)
    total_count = len(labels_arr)
    n_experimental = np.nansum(labels_arr[indices])  # count number of positive labels in subset
    temp = scores_arr[indices]

    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile  # calc EF at percentile
        ef_max = min(n_actives, sample_size) / (n_actives * percentile)
    else:
        ef = 'ND'
        ef_max = 'ND'
    return n_actives, ef, ef_max
