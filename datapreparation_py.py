# %%
import pandas as pd
from lightfm.data import Dataset
import scipy.sparse as sparse

class DataPreparation:

    def __init__(self, user_item):
        """"
        Parameters
        ----------
        user_item : Pandas Dataframe
            Dataframe with the columns ['user', 'item', 'purchases'] 
            Containing the user-item interactions
        """
        self.user_item = user_item





    def get_input_data(self, features):
        """" Function to get the input data for a LightFM model mmm

        Returns
        -------
        (interactions, weights, user_features) : tuple(coo_matrix, coo_matrix, csr_matrix)
            Data prepared for input for LightFM model
        """

        dataset_t = Dataset()
        dataset_t.fit(self.user_item.user.unique(), self.user_item.item.unique())

        interactions_matrix, weights_matrix = dataset_t.build_interactions([tuple(i) for i in self.user_item.values])
        
        user_features_sp = self._get_user_features(features)

        return (interactions_matrix, weights_matrix, user_features_sp)


    def _get_user_features(self, features):
        """" Helper function to get the user_feature values and the user_feature list

        Returns
        -------
        (user_features, user_tuple) : tuple(list, list)
            lists containing the data accordingly
        """
    
        user_features_terex = pd.read_csv('user_features_terex_new.csv', sep='|')
        available_dealers = self.user_item[['user']].drop_duplicates()
        user_features_terex = pd.merge(available_dealers, user_features_terex, left_on='user', right_on='dealer', how='left')
        
        user_features = self._aggregate_features(user_features_terex, features)

        return user_features

    def _aggregate_features(self, user_features_file, features):   
        for feature in features:
            uf_temp = user_features_file[['dealer', feature]].drop_duplicates(['dealer', feature]).dropna().set_index('dealer')
            
            user_features = []
            for c in uf_temp.columns:
                col = uf_temp[str(c)]
                for v in col.unique():
                    user_features.append(v)

            feature_list = []
            for i in range(len(uf_temp)):
                feature_list.append(list(uf_temp.iloc[i, :]))
            user_tuple = list(zip(uf_temp.index, feature_list))   

            dataset_t = Dataset()
            dataset_t.fit(self.user_item.user.unique(), self.user_item.item.unique(), user_features)

            user_features_sp = dataset_t.build_user_features(user_tuple, normalize=False)

            user_features_sp_norm = self._normalize(user_features_sp)

            if feature == features[0]:
                user_features_sp_ret = user_features_sp_norm
            else:
                user_features_sp_ret = self._add_csr(user_features_sp_ret, user_features_sp_norm)
        return user_features_sp_ret

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
            csr = sparse.hstack((csr, add[:,(add.shape[0]+i)].A.T[0][:,None]))
        return csr


