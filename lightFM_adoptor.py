# %%
import numpy as np
import multiprocessing

from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class LightFMAdaptor(MatrixFactorizationBase):
    def __init__(self, iterations=1, num_threads=0, random_state=22, *args, **kwargs):
        super(LightFMAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from lightfm import LightFM
        self.model = LightFM(random_state=random_state, *args, **kwargs)
        self.iterations = iterations
        self.num_threads = num_threads or multiprocessing.cpu_count()
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


    def evaluate(self, test_interactions, train_interactions, user_features, item_features, k=10):
        from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank

        pk = precision_at_k(model=self.model, test_interactions=test_interactions, train_interactions=train_interactions, 
        k=k, user_features=user_features, item_features=item_features).mean()
        return pk



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


    # deprecated by transpose_features!!
    def _transpose_user_features(self, user_features, model):
        if user_features is not None:
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
        else:
            num_users = model.user_embeddings.shape[0]
            return np.concatenate((model.user_embeddings, model.user_biases.reshape(num_users, 1), np.ones((num_users, 1))), axis=1).copy()
        

    # deprecated by transpose_features!!
    def _transpose_item_features(self, item_features, model):
        if item_features is not None:
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
        else:
            num_items = model.item_embeddings.shape[0]
            return np.concatenate((model.item_embeddings, model.item_biases.reshape(num_items, 1), np.ones((num_items, 1))), axis=1).copy()

    def save(self):
        pass


