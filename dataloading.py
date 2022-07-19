# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class DataLoader:

    def __init__(self):
        pass

    # integrated func to load co(d) data and return csr user_item
    def import_agco(self, file, return_type) :
        #import pandas as pd
        agco_loc = pd.read_csv('loc_agco_new.csv', sep = '|', low_memory=False)
        if file == 'CO':
            agco_cod = pd.read_csv('cod_agco_new.csv', sep = '|', low_memory=False)
            agco_cod_loc = pd.merge(agco_cod, agco_loc, left_on='supply_location_id', right_on='location_id', how='inner')
            user_item_cod_loc = agco_cod_loc[['user', 'item_id', 'requested_quantity']].groupby(by=['user', 'item_id']).sum().reset_index()
            user_item_cod_loc = user_item_cod_loc[user_item_cod_loc.requested_quantity >= 1]
            purchases_clipped_co = np.percentile(user_item_cod_loc.requested_quantity, 99)
            user_item_cod_loc['purchases'] = np.clip(user_item_cod_loc.requested_quantity, a_min=1, a_max=purchases_clipped_co)
            user_item_co = user_item_cod_loc[['user', 'item_id', 'purchases']]
            user_item_co.columns = ['user', 'item', 'purchases']
            df = user_item_co
        
        if file == 'PO':
            agco_pod = pd.read_csv('pod_agco_new.csv', sep = '|', low_memory=False)
            agco_pod_loc = pd.merge(agco_pod, agco_loc, left_on='receive_location_id', right_on='location_id', how='inner')
            user_item_pod_loc = agco_pod_loc[['user', 'item_id', 'requested_quantity']].groupby(by=['user', 'item_id']).sum().reset_index()
            user_item_pod_loc = user_item_pod_loc[user_item_pod_loc.requested_quantity >= 1]
            purchases_clipped_po = np.percentile(user_item_pod_loc.requested_quantity, 99)
            user_item_pod_loc['purchases'] = np.clip(user_item_pod_loc.requested_quantity, a_min=1, a_max=purchases_clipped_po) 
            user_item_po = user_item_pod_loc[['user', 'item_id', 'purchases']]
            user_item_po.columns = ['user', 'item', 'purchases']
            df = user_item_po
        
        if return_type == 'df':
            return df
        
        if return_type == 'csr':
            df['user'] = pd.Categorical(df.user).codes
            df['item'] = pd.Categorical(df.item).codes
            user_item_coo = coo_matrix((df.purchases, (df.user, df.item)))
            user_item_csr = user_item_coo.tocsr()
            return user_item_csr

    # function to transform the output df of import_agco to a csr matrix
    def to_csr(self, df):
        df['user'] = pd.Categorical(df.user).codes
        df['item'] = pd.Categorical(df.item).codes
        user_item_coo = coo_matrix((df.purchases, (df.user, df.item)))
        user_item_csr = user_item_coo.tocsr()
        return user_item_csr

    # func to get the number of items each user has interacted with
    def items_per_user(self, csr):
        df = pd.DataFrame({'user' : csr.tocoo().row, 'item' : csr.tocoo().col})
        df_agg = df.groupby(by=['user']).count()
        return df_agg



