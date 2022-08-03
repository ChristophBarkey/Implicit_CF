# %%
import pandas as pd
from lightfm.data import Dataset

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

    def _get_user_features(self):
        """" Helper function to get the user_feature values and the user_feature list

        Returns
        -------
        (user_features, user_tuple) : tuple(list, list)
            lists containing the data accordingly
        """
    
        user_features_terex = pd.read_csv('user_features_terex_new.csv', sep='|')
        available_dealers = self.user_item[['user']].drop_duplicates()
        user_features_terex = pd.merge(available_dealers, user_features_terex, left_on='user', right_on='dealer', how='left')
        uf_t = user_features_terex[['dealer', 'currency', 'brand']].drop_duplicates(['dealer']).dropna().set_index('dealer')
        
        # list of all distinct feature values/categories over all features
        # input for data.fit() method
        user_features = []
        for c in uf_t.columns:
            col = uf_t[str(c)]
            for v in col.unique():
                user_features.append(v)

        # list of users and tuples of respective feature values
        # input for data.build_user_features() method
        feature_list = []
        for i in uf_t.index:
            feature_list.append(list(uf_t.loc[str(i), :]))
        user_tuple = list(zip(uf_t.index, feature_list))
        return (user_features, user_tuple)

    def get_input_data(self):
        """" Function to get the input data for a LightFM model

        Returns
        -------
        (interactions, weights, user_features) : tuple(coo_matrix, coo_matrix, csr_matrix)
            Data prepared for input for LightFM model
        """
        user_features, user_tuple = self._get_user_features()
        dataset_t = Dataset()
        dataset_t.fit(self.user_item.user.unique(), self.user_item.item.unique(), user_features)

        interactions_matrix, weights_matrix = dataset_t.build_interactions([tuple(i) for i in self.user_item.values])
        user_features_sp = dataset_t.build_user_features(user_tuple, normalize= False)
        return (interactions_matrix, weights_matrix, user_features_sp)



