from __future__ import print_function

import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np


if __name__ == '__main__':
    with open('filtered_aldrich_prediction.out') as f:
        lines = f.readlines()

    pred_value_list = []
    for line in lines:
        line = line.strip().split('\t')
        old_smiles, neo_smiles, id, pred_value= line[0], line[1], line[2], float(line[3])
        pred_value_list.append(pred_value)

    pred_value_list = np.array(pred_value_list)
    min_value, max_value = np.min(pred_value_list), np.max(pred_value_list)
    print('min and max prediction value are {} and {}'.format(min_value, max_value))

    n_bins = 100
    intervals = [float(x) / n_bins for x in range(n_bins+1)]
    pred_bin_list = np.zeros(len(pred_value_list))
    print(intervals)
    print()

    for i,(left,right) in enumerate(zip(intervals[:-1], intervals[1:])):
        index = (left <= pred_value_list) & (pred_value_list < right)
        pred_bin_list[index] = left
        print('{} in [{}, {})'.format(sum(index), left, right))

    plt.hist(pred_bin_list, bins=intervals)
    plt.savefig('aldrich_prediction_distribution.png')
    plt.clf()

    plt.hist(pred_bin_list, bins=intervals)
    plt.ylim(0, 1500)
    plt.savefig('aldrich_prediction_distribution_truncated.png')
    plt.clf()