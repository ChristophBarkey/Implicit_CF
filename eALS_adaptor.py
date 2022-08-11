# %%
import numpy as np
import multiprocessing

from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

class eALSAdaptor(MatrixFactorizationBase):
    def __init__(self, num_threads=0, show_loss=False, *args, **kwargs):
        super(eALSAdaptor, self).__init__()

        # create a LightFM model using the supplied parameters
        from eals import ElementwiseAlternatingLeastSquares
        self.model = ElementwiseAlternatingLeastSquares(*args, **kwargs)

        self.show_loss = show_loss
        self.num_threads = num_threads or multiprocessing.cpu_count()

    def fit(self, user_item):
        # fit the wrapped model
        self.model.fit(user_item, 
                        show_loss=self.show_loss)
   
        # convert model attributes back to this class, so that
        # the recommend/similar_items etc calls on the base class will work
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def save(self):
        pass



