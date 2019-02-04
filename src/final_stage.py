from __future__ import print_function

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import json
from xgboost_classification import XGBoostClassification


if __name__ == '__main__':
    test_file = '../datasets/emolecules.smi'

    with open(test_file) as f:
        lines_ = f.readlines()
    lines_ = np.array(lines_)
    print('{} lines in all'.format(len(lines_)))

    FP_radius = 2
    FP_size = 1024

    handler = open('./final_stage_predicted.out', 'w')
    idx_array = np.arange(len(lines_))
    idx_splitted_array = np.array_split(idx_array, 50)
    for idx_ in idx_splitted_array:
        lines = lines_[idx_]
        print(len(idx_), '\t', len(lines))

        smiles_list = []
        fingerprints_list = []
        for idx,line in enumerate(lines):
            line_ = line.strip().split(' ')
            smiles = line_[0]
            smiles_list.append(smiles)
            mol = Chem.MolFromSmiles(smiles)
            fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_radius, nBits=FP_size).ToBitString()
            fingerprints = list(fingerprints)
            fingerprints_list.append(fingerprints)

        fingerprints_list = np.array(fingerprints_list)
        print(fingerprints_list.shape)

        config_json_file = '../config/xgboost_classification/140.json'
        weight_file = '../model_weight/xgboost_classification/xgboost_classification_140_4.pkl'

        with open(config_json_file, 'r') as f:
            task_conf = json.load(f)
        xgboost_model = XGBoostClassification(conf=task_conf)
        model = xgboost_model.load_model(weight_file)
        pred_values = model.predict_proba(fingerprints_list)[:, 1]
        print('shape: {},\tfirst 10 values: {}'.format(pred_values.shape, pred_values[:10]))
        print('over 0.1 has {}/{}'.format(sum(pred_values>0.1), len(pred_values)))

        for smiles,pred_value in zip(smiles_list, pred_values):
            print('{}\t{}'.format(smiles, pred_value), file=handler)
