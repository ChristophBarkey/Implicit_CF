# %%
import numpy as np
import multiprocessing

from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class LightFMAdaptor(MatrixFactorizationBase):
    def __init__(self, iterations=1, num_threads=0, *args, **kwargs):
        super(LightFMAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from lightfm import LightFM
        self.model = LightFM(*args, **kwargs)
        self.iterations = iterations
        self.num_threads = num_threads or multiprocessing.cpu_count()


    def fit(self, interactions, user_features=None, item_features=None, weights=None, show_progress=False):
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
   
        # convert model attributes back to this class, so that
        # the recommend/similar_items etc calls on the base class will work
        self.user_factors = self._transpose_user_features(user_features, self.model)
        self.item_factors = self._transpose_item_features(item_features, self.model)


    def _transpose_user_features(self, user_features, model):
        num_users = user_features.shape[0]
        num_features = user_features.shape[1] - user_features.shape[0]
        return_embeddings = model.user_embeddings[:num_users].copy()
        feature_embedding = model.user_embeddings[num_users:].copy()
        return_biases = model.user_biases[:num_users].copy()
        feature_biases = model.user_biases[num_users:].copy()
        for u in range(num_users):
            for i in range(num_features):
                weight = user_features[u, num_users+i]
                return_embeddings[u] += weight * feature_embedding[i]
                return_biases[u] += weight * feature_biases[i]
        return np.concatenate((return_embeddings, return_biases.reshape(num_users, 1), np.ones((num_users, 1))), axis=1).copy()

    def _transpose_item_features(self, item_features, model):
        num_items = item_features.shape[0]
        num_features = item_features.shape[1] - item_features.shape[0]
        return_embeddings = model.item_embeddings[:num_items].copy()
        feature_embedding = model.item_embeddings[num_items:].copy()
        return_biases = model.item_biases[:num_items].copy()
        feature_biases = model.item_biases[num_items:].copy()
        for u in range(num_items):
            for i in range(num_features):
                weight = item_features[u, num_items+i]
                return_embeddings[u] += weight * feature_embedding[i]
                return_biases[u] += weight * feature_biases[i]
        return np.concatenate((return_embeddings, np.ones((num_items, 1)), return_biases.reshape(num_items, 1)), axis=1).copy()
    
    def save(self):
        pass


