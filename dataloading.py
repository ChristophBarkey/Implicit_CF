# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class DataLoader:

    def __init__(self):
        pass

    # integrated func to load co(d) data and return csr user_item
    def import_data(self, OEM, file, return_type, clip=99) :
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
        clip_max = np.percentile(user_item.requested_quantity, clip)
        user_item['purchases'] = np.clip(user_item.requested_quantity, a_min=1, a_max=clip_max) 
        user_item = user_item[['user', 'item_id', 'purchases']]
        user_item.columns = ['user', 'item', 'purchases']

        
        if return_type == 'df':
            return user_item
        
        if return_type == 'csr':
            csr = self.to_csr(user_item)
            return csr

    # function to clip the purchase quantities of a user-item df
    def clip_df(self, df, clip):
        clip_max = np.percentile(df.purchases, clip)
        df['clipped'] = np.clip(df.purchases, a_min=1, a_max=clip_max) 
        df = df[['user', 'item', 'clipped']]
        df.columns = ['user', 'item', 'purchases']
        return df

    # function to merge co and po data. if user/item appears in both files, the max purchase is considered, else a full join
    def merge_co_po(self, co, po):
        full = pd.merge(co, po, on=['item', 'user'], how='outer')
        full = full.fillna(0)
        full['max'] = full[['purchases_x', 'purchases_y']].max(axis=1)
        ret = full[['user', 'item', 'max']]
        ret.columns =  ['user', 'item', 'purchases']
        return ret

    # function to transform the output df of import_agco to a csr matrix
    def to_csr(self, df_input):
        df = df_input
        df['user'] = pd.Categorical(df.user).codes
        df['item'] = pd.Categorical(df.item).codes
        user_item_coo = coo_matrix((df.purchases, (df.user, df.item)))
        user_item_csr = user_item_coo.tocsr()
        return user_item_csr
