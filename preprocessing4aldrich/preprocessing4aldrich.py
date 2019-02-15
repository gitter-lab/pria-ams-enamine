from __future__ import print_function

import json
import argparse
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='aldrich')

    args = parser.parse_args()
    target = args.target

    if target == 'aldrich':
        test_file = '../datasets/aldrich.ism'
    elif target == 'emolecules':
        test_file = '../datasets/emolecules.smi'
    else:
        raise Exception('Target Not Found!')

    with open(test_file) as f:
        lines_ = f.readlines()
    lines_ = np.array(lines_)
    print('{} lines in all'.format(len(lines_)))

    FP_radius = 2
    FP_size = 1024
    saltRemover = SaltRemover('../datasets/raw/Salts.txt')

    idx_array = np.arange(len(lines_))
    idx_splitted_array = np.array_split(idx_array, 50)
    for count,idx_ in enumerate(idx_splitted_array):
        lines = lines_[idx_]
        print(len(idx_), '\t', len(lines))

        smiles_list, fingerprints_list, datestamp_list = [], [], []
        for idx,line in enumerate(lines):
            line_ = line.strip().split(' ')
            smiles = line_[0]
            datestamp = line_[1]
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = saltRemover.StripMol(mol)
                fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_radius, nBits=FP_size).ToBitString()

                smiles_list.append(smiles)
                datestamp_list.append(datestamp)
                fingerprints_list.append(fingerprints)
            except:
                print('invalid\t', smiles)

        df = pd.DataFrame({'smiles': smiles_list, 'datestamp': datestamp_list, 'fingerprints': fingerprints_list})
        print(df.shape)
        df.to_csv('../datasets/{}/{}.csv.gz'.format(target, count), index=None, compression='gzip')
        print()
