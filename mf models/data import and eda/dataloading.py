# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


class DataLoader:

    def __init__(self):
        pass

    # integrated func to load co(d) data and return csr user_item
    def import_data(self, OEM, file, return_type, clip=100) :
        """"Function to import OEM interaction data
        Automatically removes interactions with negative quantities (after grouping)
        
        Parameters
        ----------
        OEM : str
            Identifier for OEM
        file : str
            Identifier for type of interaction, CO or PO
        return_type : str
            Identifier for data type to return
            df yields dataframe, csr yields csr_matrix
        clip : int, optional
            Option to clip the purchase quantities at the desired percentile

        Returns
        -------
        user_item_data : dataframe or csr_matrix
            user item data according to input parameters
        """
 
        if OEM == 'AGCO':

            agco_loc = pd.read_csv('loc_agco_new.csv', sep = '|', low_memory=False)
    
            if file == 'CO':
                agco_cod_18 = pd.read_csv('cod_agco_new_2018.csv', sep = '|', low_memory=False)
                agco_cod_19 = pd.read_csv('cod_agco_new_2019.csv', sep = '|', low_memory=False)
                agco_cod_20 = pd.read_csv('cod_agco_new_2020.csv', sep = '|', low_memory=False)
                agco_cod_21 = pd.read_csv('cod_agco_new_2021.csv', sep = '|', low_memory=False)
                
                agco_cod = pd.concat([agco_cod_18, agco_cod_19, agco_cod_20, agco_cod_21])
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
        user_item = user_item[['user', 'item_id', 'requested_quantity']]
        user_item.columns = ['user', 'item', 'purchases']
        if clip < 100:
            user_item = self.clip_df(user_item, clip)
        
        if return_type == 'df':
            return user_item
        
        if return_type == 'csr':
            csr = self.to_csr(user_item)
            return csr

    # function to clip the purchase quantities of a user-item df
    def clip_df(self, df, clip):
        ret = df.copy()
        clip_max = np.percentile(df.purchases, clip)
        ret['purchases'] = np.clip(ret.purchases, a_min=1, a_max=clip_max) 
        return ret

    # function to transform purchase values by log
    def log_scale_df(self, df, epsilon, alpha=1):
        """"Function to scale the purchase quantities in a logarithmic way.
        Oriented on the log scaling scheme from Hu et. al. 2008

        Parameters
        ----------
        df : dataframe
            Pandas dataframe with column 'purchases' to be scaled
        epsilon : float
            Scaling parameter epsilon. Suggested 0.01
        alpha : float, optional
            Scaling parameter alpha, mutipliying the scaled values

        Returns
        -------
        df : dataframe
            Copy of input dataframe, with scaled purchases
        """
        ret = df.copy()
        ret['purchases'] = alpha * np.log(1 + (df.purchases/epsilon))
        return ret

    # function to transform the output df of import_agco to a csr matrix
    def to_csr(self, df):
        """"Function to transform dataframe of user item data to csr_data
        The users and items will be transformed to running integers, starting with 0

        Parameters
        ----------
        df : dataframe
            Pandas dataframe with user item data
            Columns: user, item, purchases

        Returns
        -------
        csr : csr_matrix
            Same user item data in sparse csr format
        """     
        ret = df.copy()
        ret['user'] = pd.Categorical(ret.user).codes
        ret['item'] = pd.Categorical(ret.item).codes
        user_item_coo = coo_matrix((ret.purchases, (ret.user, ret.item)))
        user_item_csr = user_item_coo.tocsr()
        return user_item_csr

    # function to remove lines with items that were only bought by less or equal than n users; 1 per default
    def remove_low_interact_items(self, df, n=1):
        """"Function to remove interactions where the items were only bought by n or less users

        Parameters
        ----------
        df : dataframe
            Pandas dataframe with column user item data
        n : int, optional
            Interactions with number of users <= n will be removed
            Default 1

        Returns
        -------
        df : dataframe
            Copy of input dataframe, with removed lines
        """        
        ret = df.copy()
        duplicate_items = ret.item.value_counts() > n
        ret_filtered = ret[ret.item.isin(duplicate_items[duplicate_items].index)]
        return ret_filtered
