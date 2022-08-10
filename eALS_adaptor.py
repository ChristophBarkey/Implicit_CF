# %%
import numpy as np
import multiprocessing

from implicit.cpu import MatrixFactorizationBase

class eALSAdaptor(MatrixFactorizationBase):
    def __init__(self, *args, **kwargs):
        super(eALSAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from eals import ElementwiseAlternatingLeastSquares
        self.model = ElementwiseAlternatingLeastSquares(*args, **kwargs)

        self.show_loss = True

    def fit(self, user_items):
        # fit the wrapped model
        self.model.fit(user_items, 
                       show_loss=self.show_loss)
   
        # convert model attributes back to this class, so that
        # the recommend/similar_items etc calls on the base class will work
        users, items  = user_items.shape
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors,



