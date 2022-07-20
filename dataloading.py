# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class DataLoader:

    def __init__(self):
        pass

    # integrated func to load co(d) data and return csr user_item
    def import_data(self, OEM, file, return_type) :
        #import pandas as pd
 
        if OEM == 'AGCO':

            agco_loc = pd.read_csv('loc_agco_new.csv', sep = '|', low_memory=False)
    
            if file == 'CO':
                agco_cod = pd.read_csv('cod_agco_new.csv', sep = '|', low_memory=False)
                copod_loc = pd.merge(agco_cod, agco_loc, left_on='supply_location_id', right_on='location_id', how='inner')
        
            
            if file == 'PO':
                agco_pod = pd.read_csv('pod_agco_new.csv', sep = '|', low_memory=False)
                copod_loc = pd.merge(agco_pod, agco_loc, left_on='receive_location_id', right_on='location_id', how='inner')
                
        if OEM == 'TEREX':
           
            terex_loc = pd.read_csv('loc_terex_new.csv', sep = '|', low_memory=False)
    
            if file == 'CO':
                terex_cod = pd.read_csv('cod_terex_new.csv', sep = '|', low_memory=False)
                copod_loc = pd.merge(terex_cod, terex_loc, left_on='supply_location_id', right_on='location_id', how='inner')
        
            
            if file == 'PO':
                terex_pod = pd.read_csv('pod_terex_new.csv', sep = '|', low_memory=False)
                copod_loc = pd.merge(terex_pod, terex_loc, left_on='receive_location_id', right_on='location_id', how='inner') 


        user_item = copod_loc[['user', 'item_id', 'requested_quantity']].groupby(by=['user', 'item_id']).sum().reset_index()
        user_item = user_item[user_item.requested_quantity >= 1]
        clip_max = np.percentile(user_item.requested_quantity, 99)
        user_item['purchases'] = np.clip(user_item.requested_quantity, a_min=1, a_max=clip_max) 
        user_item = user_item[['user', 'item_id', 'purchases']]
        user_item.columns = ['user', 'item', 'purchases']

        
        if return_type == 'df':
            return user_item
        
        if return_type == 'csr':
            csr = self.to_csr(user_item)
            return csr

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



