"""
    Usage:
        python filter_enamine.py
"""

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, './rd_filters/')
from rd_filters import *

"""
    Computes closest actives_df compounds to prosp_df compounds.
"""
def get_sim_df(actives_df, prosp_df, dataset):
    X_train_actives = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in actives_df['1024 MorganFP Radius 2']]).astype(float)
    X_prosp = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in prosp_df['1024 MorganFP Radius 2']]).astype(float)
    train_prosp_tandist = pairwise_distances(X_prosp, X_train_actives, metric='jaccard')
    sim_info = []
    for i in range(train_prosp_tandist.shape[0]):
        prosp_id = prosp_df['Index ID'].iloc[i]; 
        closest_active_idx = np.argmin(train_prosp_tandist[i,:]); 
        dist = train_prosp_tandist[i,closest_active_idx]
        ctrid = actives_df['Index ID'].iloc[closest_active_idx];
        
        
        sim_info.append((prosp_id, ctrid, dist))
        
    sim_df = pd.DataFrame(data=sim_info, 
                          columns=['Index ID', 'Closest {} Active ID'.format(dataset), 'Closest {} Active TanDist'.format(dataset)])
    return sim_df

if __name__ ==  '__main__':
    ################################### read in data and compute nearest actives ###############################################
    top_real_clustering = pd.read_csv('../datasets/real_costs_clustered_v3.csv.gz')
    train_df = pd.read_csv('../datasets/folds/training_df_single_fold_with_clustering.csv.gz')
    ams_df = pd.read_csv('../datasets/ams_order_results.csv.gz')

    train_actives = train_df[train_df['PriA-SSB AS Activity'] == 1]
    sim_train_df = get_sim_df(train_actives, top_real_clustering, 'Train')

    ams_actives = ams_df[ams_df['Hit'] == 1]
    sim_ams_df = get_sim_df(ams_actives, top_real_clustering, 'AMS')

    top_real_clustering = top_real_clustering.merge(sim_train_df, on='Index ID')
    top_real_clustering = top_real_clustering.merge(sim_ams_df, on='Index ID')
    
    alert_file_name = './rd_filters/data/alert_collection.csv'
    rules_file_name = './rd_filters/data/rules.json'
    rf = RDFilters(alert_file_name)
    rules_file_path = get_config_file(rules_file_name, "FILTER_RULES_DATA")
    rule_dict = read_rules(rules_file_path)
    rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
    rule_str = " and ".join(rule_list)
    print(f"Using alerts from {rule_str}")
    rf.build_rule_list(rule_list)

    input_data = top_real_clustering[["rdkit SMILES", "idnumber"]].values.tolist()

    start_time = time.time()
    p = Pool(1)
    filter_res = list(p.map(rf.evaluate, input_data))
    rd_df = pd.DataFrame(filter_res, columns=["SMILES", "idnumber", "RD_FILTER", "MW", "LogP", "HBD", "HBA", "TPSA"])
    filter_binary = np.zeros((rd_df.shape[0],), dtype='uint8')
    for index, row in rd_df.iterrows():
        if row['RD_FILTER'] == "OK":
            filter_binary[index] = 1
    rd_df["RD_FILTER"] = filter_binary
    rd_df = rd_df[["idnumber", "RD_FILTER", "LogP"]]

    top_real_clustering = top_real_clustering.merge(rd_df, on='idnumber')
    # aggregate costs for 1-50
    top_real_clustering['Price'] = top_real_clustering[['Price for 11-50 cmpds', 'Price for 40-49 cmpds']].sum(axis=1).values
    
    des_cols = ['ID Enamine', 
            'REAL SMILES', 'rdkit SMILES', 'PAINS Filter', 
               'TB_0.4 ID', 'Hit', 'Is Train cluster?',
               'Is Train active cluster?', 'Is AMS cluster?',
               'Is AMS active cluster?', 'rf_preds',
               'Price', 'Status', 'Delivery term, weeks',
               'Closest Train Active ID', 'Closest Train Active TanDist',
               'Closest AMS Active ID', 'Closest AMS Active TanDist', '1024 MorganFP Radius 2',
               "RD_FILTER", "MW", "LogP", "HBD", "HBA", "TPSA"]
    df = top_real_clustering[des_cols].sort_values('rf_preds', ascending=False)
    df['rf_rank'] = df['rf_preds'].rank(method='first', ascending=False)
    
    
    ################################### Apply filters ###############################################
    # apply filters
    print('Shape: {}'.format(df.shape[0]))
    # 1. get cpds in clusters that don't exist in train or ams ACTIVE clusters
    prune_df = df[(df['Is Train active cluster?'] == 0) & (df['Is AMS active cluster?'] == 0)]
    print('1. Shape: {}'.format(prune_df.shape[0]))

    # 2. pass PAINS and RD_FILTERS filter
    prune_df = prune_df[(prune_df['PAINS Filter'] == 1) * (prune_df['RD_FILTER'] == 1)]
    print('2. Shape: {}'.format(prune_df.shape[0]))

    # 3. take only cpds that are from from tain and ams actives
    tandist_thresh=0.35
    prune_df = prune_df[(prune_df['Closest Train Active TanDist'] >= tandist_thresh) & (prune_df['Closest AMS Active TanDist'] >= tandist_thresh)]
    print('3. Shape: {}'.format(prune_df.shape[0]))

    # get highest prediction from each cluster
    prune_df = prune_df.drop_duplicates(subset='TB_0.4 ID', keep='first').reset_index(drop=True)
    print('4. Shape: {}'.format(prune_df.shape[0]))
    
    
    num_cpds = 100
    cpds_to_select = [0] # first select cpd with highest rf_rank

    X_prosp = np.vstack([np.fromstring(x, 'u1') - ord('0') for x in prune_df['1024 MorganFP Radius 2']]).astype(float)
    for i in range(1, 100):
        x = X_prosp[cpds_to_select,:]
        remaining_cpds = np.setdiff1d(np.arange(X_prosp.shape[0]), cpds_to_select)
        y = X_prosp[remaining_cpds,:]
        tandist = pairwise_distances(y, x, metric='jaccard')
        farthest_idx = np.argmax(tandist.mean(axis=1)); 

        cpds_to_select.append(remaining_cpds[farthest_idx])

    final_list = prune_df.iloc[cpds_to_select,:]
    final_list = final_list.sort_values('rf_preds', ascending=False)