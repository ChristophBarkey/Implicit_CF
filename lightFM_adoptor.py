# %%
import numpy as np
import multiprocessing

from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class LightFMAdaptor(MatrixFactorizationBase):
    def __init__(self, epochs=20, num_threads=0, *args, **kwargs):
        super(LightFMAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from lightfm import LightFM
        self.model = LightFM(*args, **kwargs)

        self.epochs = epochs
        self.show_progress = True
        self.num_threads = num_threads or multiprocessing.cpu_count()

    def fit(self, item_users):
        # fit the wrapped model
        self.model.fit(item_users.T.tocoo(), 
                       num_threads=self.num_threads,
                       epochs=self.epochs,
                       verbose=self.show_progress)
   
        # convert model attributes back to this class, so that
        # the recommend/similar_items etc calls on the base class will work
        items, users = item_users.shape
        self.user_factors = np.concatenate((self.model.user_embeddings,
                                            self.model.user_biases.reshape(users, 1),
                                            np.ones((users, 1))), axis=1).copy()
        self.item_factors = np.concatenate((self.model.item_embeddings,
                                            np.ones((items, 1)),
                                            self.model.item_biases.reshape(items, 1)),
                                            axis=1).copy()


