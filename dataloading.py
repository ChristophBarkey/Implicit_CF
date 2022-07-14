# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class DataLoader:

    def __init__(self):
        pass

    # integrated func to load co(d) data and return csr user_item
    def import_agco(self, file, clip) :
        locations = pd.read_csv('locations_agco_new.csv', sep='|', low_memory=False)
        if file == 'CO':
            co = pd.read_csv('co_agco_new.csv', sep = '|', low_memory=False)
            cd = pd.read_csv('cd_agco_new.csv', sep = '|', low_memory=False)
            cod = pd.merge(co, cd, on='co_id', how='inner')
            cod_loc = pd.merge(cod, locations, left_on='supply_location_id', right_on='location_id', how='left')
            dealer_items = cod_loc[['group1', 'item_id', 'requested_quantity']]
        
        if file == 'PO':
            po = pd.read_csv('po_agco_new.csv', sep = '|', low_memory=False)
            pd = pd.read_csv('pd_agco_new.csv', sep = '|', low_memory=False)
            pod = pd.merge(po, pd, on='po_id', how='inner')
            pod_loc = pd.merge(pod, locations, left_on='receive_location_id', right_on='location_id', how='left')
            dealer_items = pod_loc[['group1', 'item_id', 'requested_quantity']]
        
        user_item = dealer_items.groupby(by=['group1', 'item_id']).sum().reset_index()
        user_item.columns = ['user', 'item', 'purchases']
        user_item = user_item.loc[user_item.purchases > 0]
        if clip < 100 :
            user_item['purchases'] = np.clip(user_item.purchases, a_min=1, a_max=np.percentile(user_item.purchases, clip))
        user_item['user'] = pd.Categorical(user_item.user).codes
        user_item['item'] = pd.Categorical(user_item.item).codes
        user_item_coo = coo_matrix((user_item.purchases, (user_item.user, user_item.item)))
        user_item_csr = user_item_coo.tocsr()
        return user_item_csr

    # func to get the number of items each user has interacted with
    def items_per_user(self, csr):
        df = pd.DataFrame({'user' : csr.tocoo().row, 'item' : csr.tocoo().col})
        df_agg = df.groupby(by=['user']).count()
        return df_agg



