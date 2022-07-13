# %%
from cv_py import CrossValidation

import pandas as pd
import numpy as np
from itertools import product
import implicit

class CrossValidationiALS(CrossValidation):
    
    def __init__(self, user_item, k):
        CrossValidation.__init__(self, user_item, k)

    def hyperp_tuning(self, test, train, param_space, eval):
        keys, values = zip(*param_space.items())
        result = [dict(zip(keys, p)) for p in product(*values)]
        first_iter = True
        for r in result:
            model = implicit.als.AlternatingLeastSquares(factors=r['factors'], regularization=r['regularization'], alpha=r['alpha'])
            if eval == 'cv':
                res = self.k_fold_eval(test, train, model, return_type='mean')
            if eval == 'split':
                res = self.evaluate_model(model, train, test, 10)

            if first_iter == True:
                metrics_frame = res
                first_iter = False
            else:
                metrics_frame = pd.concat((metrics_frame, res), axis=0)
        param_df = pd.DataFrame(result)
        ret = pd.concat((param_df.reset_index(drop=True), metrics_frame.reset_index(drop=True)), axis=1)
        return ret
    


