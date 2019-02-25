from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
from function import read_merged_data, extract_feature_and_label, reshape_data_into_2_dim
from evaluation import roc_auc_single, precision_auc_single
from util import output_classification_result

"""
    Similarity Baseline computes the tanimoto similarity of a molecule to the actives
    in the training set. The molecule's prediction is then set to the maximum tanimoto
    similarity.
"""
class SimilarityBaseline:
    def __init__(self, conf):
        self.conf = conf
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        
        self.X_train_actives = None
        return
    
    def fit(self, X_train, y_train):
        actives_indices = np.where(y_train == 1)[0]
        self.X_train_actives = X_train[actives_indices]
        return
    
    @staticmethod
    def _tanimoto_similarity(x1, x2):
        x1 = x1.astype(bool)
        x2 = x2.astype(bool)
        tan_sim = np.sum(np.bitwise_and(x1, x2)) / np.sum(np.bitwise_or(x1, x2))
        return tan_sim
    
    def _baseline_pred(self, X_data):
        y_preds = np.zeros(shape=(X_data.shape[0], 1))
        ts_tmp = np.zeros(shape=(self.X_train_actives.shape[0],))
        for xi, x_fps in enumerate(X_data):
            for tsi, active_fps in enumerate(self.X_train_actives):
                ts_tmp[tsi] = SimilarityBaseline._tanimoto_similarity(x_fps, active_fps)
            y_preds[xi] = np.max(ts_tmp)
        return y_preds

    def train_and_predict(self, X_train, y_train, X_test, y_test, weight_file):
        self.fit(X_train, y_train)
        
        y_pred_on_train = self._baseline_pred(X_train)
        if X_test is not None:
            y_pred_on_test = self._baseline_pred(X_test)
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test,
                                     EF_ratio_list=self.EF_ratio_list)

        self.save_model(weight_file)
        return

    def predict_with_existing(self, X_data, weight_file):
        self.load_model(weight_file)
        y_pred = self._baseline_pred(X_data)
        return y_pred

    def eval_with_existing(self, X_train, y_train, X_test, y_test, weight_file):
        self.load_model(weight_file)

        y_pred_on_train = self._baseline_pred(X_train)
        if X_test is not None:
            y_pred_on_test = self._baseline_pred(X_test)
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test,
                                     EF_ratio_list=self.EF_ratio_list)
        return

    def save_model(self, weight_file):
        np.save(weight_file, self.X_train_actives)
        return

    def load_model(self, weight_file):
        self.X_train_actives = np.load(weight_file)
        return 


def demo_similarity_baseline():
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    conf = {
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'label_name_list': ['Keck_Pria_AS_Retest']
    }

    label_name_list = conf['label_name_list']
    print('label_name_list ', label_name_list)

    test_index = 0
    complete_index = np.arange(K)
    train_index = np.where(complete_index != test_index)[0]
    train_file_list = file_list[train_index]
    test_file_list = file_list[test_index:test_index + 1]

    print('train files ', train_file_list)
    print('test files ', test_file_list)

    train_pd = read_merged_data(train_file_list)
    test_pd = read_merged_data(test_file_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)

    task = SimilarityBaseline(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)
    
    # test that tanimoto similarity is computer correctly
    m1 = AllChem.MolFromSmiles('C=CCN(C(=O)CNS(=O)(=O)c1ccccc1)C(C(=O)NCCOC)c1cccnc1')
    m2 = AllChem.MolFromSmiles('CCOC(=O)C1CCCN(C(=O)Cc2ccc(OC)c(OC)c2)CC1')
    fps1 = AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
    fps2 = AllChem.GetMorganFingerprintAsBitVect(m2,2,nBits=1024)
    x1 = np.array([int(x) for x in fps1.ToBitString()])
    x2 = np.array([int(x) for x in fps2.ToBitString()])
    assert DataStructs.TanimotoSimilarity(fps1, fps2) == task._tanimoto_similarity(x1, x2)
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', action='store', dest='weight_file', required=True)
    given_args = parser.parse_args()
    weight_file = given_args.weight_file

    # specify dataset
    K = 5
    directory = '../datasets/keck_pria_lc/{}.csv'
    file_list = []
    for i in range(K):
        file_list.append(directory.format(i))
    file_list = np.array(file_list)

    demo_similarity_baseline()