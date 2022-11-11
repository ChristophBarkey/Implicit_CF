# %%
import numpy as np
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class LightFMAdaptor(MatrixFactorizationBase):

    # Class that contains wrapper functions for lightFM models to be evaluated with functions of the class implicit
    def __init__(self, iterations=1, random_state=22, *args, **kwargs):
        super(LightFMAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from lightfm import LightFM
        self.model = LightFM(random_state=random_state, *args, **kwargs)
        self.iterations = iterations
        self.num_threads = 1
        self.random_state = random_state


    def fit(self, interactions, user_features=None, item_features=None, weights=None, show_progress=False, transform_factors=True):
        # fit the wrapped model
        if weights is not None:
            sample_weight = weights.tocoo()
        else:
            sample_weight = weights
        self.model.fit(interactions=interactions, 
                        user_features = user_features, 
                        item_features = item_features, 
                        sample_weight = sample_weight,
                        num_threads=self.num_threads,
                        epochs=self.iterations,
                        verbose=show_progress)
   
        if transform_factors:
            # convert model attributes back to this class, so that
            # the recommend/similar_items etc calls on the base class will work
            self.user_factors = self._transpose_features(user_features, self.model.user_embeddings, self.model.user_biases, 'user')
            self.item_factors = self._transpose_features(item_features, self.model.item_embeddings, self.model.item_biases, 'item')

    # evluation function using the lightfm functions, to assure equal results of the implicit function and lightfm function
    # Not used in thesis, only for testing this Class
    def evaluate(self, test_interactions, train_interactions, user_features, item_features, k=10):
        from lightfm.evaluation import precision_at_k
        pk = precision_at_k(model=self.model, test_interactions=test_interactions, train_interactions=train_interactions, 
        k=k, user_features=user_features, item_features=item_features).mean()
        return pk


    # Function to aggregate user, item and potential feature factors to the equivalent user and item factors
    def _transpose_features(self, features, embeddings, biases, instance):
        if features is not None:
            num_ui = features.shape[0]
            return_embeddings = embeddings[:num_ui].copy()
            feature_embedding = embeddings[num_ui:].copy()
            return_biases = biases[:num_ui].copy()
            feature_biases = biases[num_ui:].copy()
            
            for ui in range(num_ui):
                weights_ui = features[ui, num_ui:].toarray().T[:, 0]
                emb_arr = weights_ui.dot(feature_embedding)
                bias_arr = weights_ui.dot(feature_biases)
                return_embeddings[ui] += emb_arr
                return_biases[ui] += bias_arr
                
            if instance == 'user':
                return np.concatenate((return_embeddings, return_biases.reshape(num_ui, 1), np.ones((num_ui, 1))), axis=1).copy()
            if instance == 'item':
                return np.concatenate((return_embeddings, np.ones((num_ui, 1)), return_biases.reshape(num_ui, 1)), axis=1).copy()          
        else:
            num_ui = embeddings.shape[0]
            if instance == 'user':
                return np.concatenate((embeddings, biases.reshape(num_ui, 1), np.ones((num_ui, 1))), axis=1).copy()
            if instance == 'item':
                return np.concatenate((embeddings, np.ones((num_ui, 1)), biases.reshape(num_ui, 1)), axis=1).copy()



