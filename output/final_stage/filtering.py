from __future__ import print_function

import pandas as pd


def fetch_training_molecules():
    smiles_list = []
    count = 0
    for i in range(10):
        data_file = '../../datasets/keck_pria/fold_{}.csv'.format(i)
        df = pd.read_csv(data_file)
        smiles_list.extend(df['rdkit SMILES'].tolist())

        count += df.shape[0]

    smiles_list = set(smiles_list)
    print('{} uniques out of {}'.format(len(smiles_list), count))
    return smiles_list


if __name__ == '__main__':
    training_smiles_list = fetch_training_molecules()
    predicted_file = 'aldrich_prediction.out'

    with open(predicted_file) as f:
        lines = f.readlines()

    handler = open('filtered_aldrich_prediction.out', 'w')

    count = 0
    for line in lines:
        line = line.strip().split('\t')
        old_smiles, neo_smiles, id, pred_value= line[0], line[1], line[2], line[3]
        if neo_smiles in training_smiles_list:
            print('Duplicate SMILES: {}'.format(neo_smiles))
            count += 1
        else:
            print('{}\t{}\t{}\t{}'.format(old_smiles, neo_smiles, id, pred_value), file=handler)

    print('{} duplicates out of {}'.format(count, len(lines)))
