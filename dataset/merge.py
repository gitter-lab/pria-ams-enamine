import pandas as pd


whole_pd = pd.read_csv('{}.csv.gz'.format(0))
print whole_pd.shape

for i in range(1, 10):
    data_pd = pd.read_csv('{}.csv.gz'.format(i))
    print data_pd.shape
    whole_pd = whole_pd.append(data_pd)

print 'complete\t', whole_pd.shape
whole_pd.to_csv('keck_pria.csv.gz', compression='gzip', index=None)