import pandas as pd


if __name__ == '__main__':
    for i in range(10):
        dataframe = pd.read_csv('../keck_pria/fold_{}.csv'.format(i), nrows=1000)
        dataframe.to_csv('fold_{}.csv'.format(i), index=None)