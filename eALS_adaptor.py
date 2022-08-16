# %%
import numpy as np
import multiprocessing

from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class eALSAdaptor(MatrixFactorizationBase):
    def __init__(self, num_threads=0, *args, **kwargs):
        super(eALSAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from eals import ElementwiseAlternatingLeastSquares
        self.model = ElementwiseAlternatingLeastSquares(*args, **kwargs)

        self.num_threads = num_threads or multiprocessing.cpu_count()

    def fit(self, user_item, show_loss=False):
        # fit the wrapped model
        self.model.fit(user_item, show_loss=show_loss)
   
        # convert model attributes back to this class, so that
        # the recommend/similar_items etc calls on the base class will work
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    # just to avoid error: cannot instantiate abstract class with abstract method save
    def save(self):
        pass



