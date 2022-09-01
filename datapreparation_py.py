# %%
import pandas as pd
import numpy as np
from lightfm.data import Dataset
import scipy.sparse as sparse
from scipy.sparse import coo_matrix

class DataPreparation:

    #def __init__(self, user_item):
    #    """"
    #    Parameters
    #    ----------
    #    user_item : Pandas Dataframe
    #        Dataframe with the columns ['user', 'item', 'purchases'] 
    #        Containing the user-item interactions
    #    """
    #    self.user_item = user_item


    # NEW TEST
    def __init__(self, user_item):
        """"
        Parameters
        ----------
        user_item : Pandas Dataframe
            Dataframe with the columns ['user', 'item', 'purchases'] 
            Containing the user-item interactions


        Attributes
        ----------
        user_ite_full : Pandas df
            user_item df extended with extra columns being user_codes and item_codes.
            These codes are used for generating csr interaction matrix.
            Also used to get the correct user and item order as used in the csr data.
            These orders are necessary to build user and item features.
        users : Pandas Series
            Dealers as in user_item but unique values and ordered by user_codes.
            Used for building user features later. 
            Specifically to achieve the same mapping for the features as for the interaction matrix.
        items : Pandas Sereis
            Items as in  user_items but unique values and orderd by item_codes.
            Same purpose as the users attr. 
        """
        self.user_item = user_item
        self.user_item_full = self._get_full_user_item()
        self.users = self.user_item_full[['user', 'user_codes']].drop_duplicates(['user', 'user_codes']).sort_values(['user_codes']).user
        self.items = self.user_item_full[['item', 'item_codes']].drop_duplicates(['item', 'item_codes']).sort_values(['item_codes']).item


    # helper function to build the user_item_full frame
    def _get_full_user_item(self):
        user_item_full = self.user_item.copy()
        user_item_full['user_codes'] = pd.Categorical(user_item_full.user).codes
        user_item_full['item_codes'] = pd.Categorical(user_item_full.item).codes
        return user_item_full


    def get_interaction_data(self):
        """"
        Returns
        -------
        (interactions, weights) : (csr_matrix, csr_matrix)
            interaction_matrix marks all user item interactions as 1
            Weights contains the actual purchase quantities
        """
        df = self.user_item_full.copy()
        weights_coo = coo_matrix((df.purchases, (df.user_codes, df.item_codes)))
        weights_csr = weights_coo.tocsr()
        interactions_csr = weights_csr.sign()
        return (interactions_csr, weights_csr)


    def get_feature_data(self, OEM='TEREX', user_features=None, item_features=None):
        """"
        Parameters
        ----------
        OEM : str
            OEM identifier
        user_features : list, optional
            list of user features to include.
            Must be of ['country', 'brand', 'currency', 'dealer_size'] OEM#1
            Must be of ['country', 'brand', 'currency'] OEM#2

            
        item_features : list, optional
            list of item_features to include
            Must be of ['group2', 'movement_code', 'cost_class']     

        Returns
        -------
        [user_features, item_features] : list(csr_matrix, csr_matrix)
            user_feature matrix and item_feature matrix for lightfm model
        """
        return_list = []
        
        if user_features is not None:
            user_features_sp = self._get_user_features(OEM, user_features)
            return_list.append(user_features_sp)

        if item_features is not None:
            item_features_sp = self._get_item_features(OEM, item_features)
            return_list.append(item_features_sp)

        return return_list

    # OLD METHOD --> USE get_interaction_data + get_feature_data
    #def get_input_data(self, OEM, user_features=None, item_features=None):
    #    """" Function to get the input data for a LightFM model
    #    Parameters
    #    ----------
    #    user_features : list, optional
    #        list of user features to include.
    #        Must be of ['country', 'brand', 'currency']
    #    item_features : list, optional
    #        list of item_features to include
    #        Must be of ['group2', 'movement_code', 'cost_class']
    #    Returns
    #    -------
    #    (interactions, weights, user_features) : tuple(coo_matrix, coo_matrix, csr_matrix)
    #        Data prepared for input for LightFM model
    #    """

    #    dataset_t = Dataset()
    #    dataset_t.fit(self.user_item.user.unique(), self.user_item.item.unique())

    #    interactions_matrix, weights_matrix = dataset_t.build_interactions([tuple(i) for i in self.user_item.values])

    #    return_list = [interactions_matrix, weights_matrix]
        
    #    if user_features is not None:
    #        user_features_sp = self._get_user_features(OEM, user_features)
    #        return_list.append(user_features_sp)

    #    if item_features is not None:
    #        item_features_sp = self._get_item_features(OEM, item_features)
    #        return_list.append(item_features_sp)

    #    return return_list


    def _get_user_features(self, OEM, features):
        """" Helper function to get the user_feature values and the user_feature list

        Returns
        -------
        (user_features, user_tuple) : tuple(list, list)
            lists containing the data accordingly
        """
        if OEM == 'AGCO':
            user_features_file = pd.read_csv('user_features_agco_new.csv', sep='|')
            user_features_file = self._map_dealer_size(user_features_file)
        if OEM == 'TEREX':
            user_features_file = pd.read_csv('user_features_terex_new.csv', sep='|')
        
        available_dealers = self.user_item[['user']].drop_duplicates()
        user_features_filtered = pd.merge(available_dealers, user_features_file, left_on='user', right_on='dealer', how='left')
        
        user_features_ret = self._aggregate_features(user_features_filtered, features, 'user')

        return user_features_ret

    def _get_item_features(self, OEM, features):
        """" Helper function to get the user_feature values and the user_feature list

        Returns
        -------
        (user_features, user_tuple) : tuple(list, list)
            lists containing the data accordingly
        """
        if OEM == 'AGCO':
            items_file = pd.read_csv('items_agco_new.csv', sep='|')
            skus_file = pd.read_csv('skus_agco_new.csv', sep='|', low_memory=False)
            skus_file = self._map_cps(skus_file)
        if OEM == 'TEREX':
            items_file = pd.read_csv('items_terex_new.csv', sep='|')
            skus_file = pd.read_csv('skus_terex_new.csv', sep='|', low_memory=False)
        
        available_items = self.user_item[['item']].drop_duplicates()
        
        items_filtered = pd.merge(available_items, items_file, left_on='item', right_on='item_id', how='left')
        

        item_sku_features = pd.merge(items_filtered, skus_file, on='item_id', how='left')
        
        item_features_ret = self._aggregate_features(item_sku_features, features, 'item')

        return item_features_ret

    def _aggregate_features(self, features_file, features, instance):   
        for feature in features:
            if instance == 'user':
                uif_temp = features_file[['dealer', feature]].dropna().set_index('dealer')
            if instance == 'item':
                uif_temp = features_file[['item', feature]].dropna().set_index('item')
            
            ui_features = self._gather_features(uif_temp)

            ui_tuple = self._get_feature_tuple(uif_temp)

            dataset_t = Dataset()
            
            if instance == 'user':
                # old version: users and items as in file. But different order as in csr interaction matrix!
                #dataset_t.fit(self.user_item.user.unique(), self.user_item.item.unique(), user_features = ui_features)

                # users and items orderd as in interaction data, as these are used for index mapping for feature matrices
                dataset_t.fit(self.users, self.items, user_features = ui_features)
                ui_features_sp = dataset_t.build_user_features(ui_tuple, normalize=False)
            
            if instance == 'item':
                # old version: users and items as in file. But different order as in csr interaction matrix!
                #dataset_t.fit(self.user_item.user.unique(), self.user_item.item.unique(), item_features = ui_features)

                # users and items orderd as in interaction data, as these are used for index mapping for feature matrices
                dataset_t.fit(self.users, self.items, item_features = ui_features)
                ui_features_sp = dataset_t.build_item_features(ui_tuple, normalize=False)


            ui_features_sp_norm = self._normalize(ui_features_sp)

            if feature == features[0]:
                ui_features_sp_ret = ui_features_sp_norm
            else:
                ui_features_sp_ret = self._add_csr(ui_features_sp_ret, ui_features_sp_norm)
        return ui_features_sp_ret
    
    def _gather_features(self, feature_file):
        features = []
        for c in feature_file.columns:
            col = feature_file[str(c)]
            for v in col.unique():
                features.append(v)
        return features

    def _get_feature_tuple(self, feature_file):
        feature_list = []
        for i in range(len(feature_file)):
            feature_list.append(list(feature_file.iloc[i, :]))
        feature_tuple = list(zip(feature_file.index, feature_list))   
        return feature_tuple

    def _normalize(self, csr):
        lil = csr.tolil()
        for i in range(csr.shape[0]):
            array = csr[i, csr.shape[0]:].toarray()
            if array.sum() == 0:
                norm_array = array
            else:
                norm_array = array / array.sum()
            lil[i, csr.shape[0]:] = norm_array
        return lil.tocsr()

    def _add_csr(self, csr, add):
        for i in range((add.shape[1]-add.shape[0])):
            csr_ret = sparse.hstack((csr, add[:,(add.shape[0]+i)].A.T[0][:,None]), format='csr')
        return csr_ret

    def _map_dealer_size(self, user_features):
        user_features_temp = user_features.copy()
        user_features_temp['item_count'] = user_features_temp.item_count.fillna(0)
        col = user_features_temp.item_count
        bins = [col.min(), np.percentile(col, 25), np.percentile(col, 50), np.percentile(col, 75), col.max()]
        names = ['1', '2', '3', '4']
        user_features_temp['dealer_size'] = pd.cut(user_features_temp['item_count'], bins=bins, labels=names)
        return user_features_temp

    def _map_cps(self, skus):
        skus_temp = skus.copy()
        skus_temp['comparative_planning_score'] = skus_temp.comparative_planning_score.fillna(0)
        col = skus_temp.comparative_planning_score
        bins = [col.min(), np.percentile(col, 25), np.percentile(col, 50), np.percentile(col, 75), col.max()]
        names = ['1', '2', '3', '4']
        skus_temp['cps_category'] = pd.cut(skus_temp['comparative_planning_score'], bins=bins, labels=names)
        return skus_temp


