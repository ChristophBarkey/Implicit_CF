# %%
import numpy as np
import multiprocessing

from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class eALSAdaptor(MatrixFactorizationBase):
    def __init__(self, *args, **kwargs):
        super(eALSAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from eals import ElementwiseAlternatingLeastSquares
        self.model = ElementwiseAlternatingLeastSquares(*args, **kwargs)

        self.show_loss = False

    def fit(self, item_users):
        # fit the wrapped model
        self.model.fit(item_users, 
                       show_loss=self.show_loss)
   
        # convert model attributes back to this class, so that
        # the recommend/similar_items etc calls on the base class will work
        items, users   = item_users.shape
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors,

    def save(self):
        pass



