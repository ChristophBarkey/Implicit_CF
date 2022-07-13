# %%
from cv_py import CrossValidation

import pandas as pd
import numpy as np
from itertools import product
import implicit

class CrossValidationiALS(CrossValidation):
    """"Class extending CrossValidation for model-specific hyperparameter tuning method
    """
    def __init__(self, user_item, k):
        CrossValidation.__init__(self, user_item, k)

    def hyperp_tuning(self, test, train, param_space, eval):
        """" Hyperparameter tuning method for iALS models

        Parameters
        ----------
        test : dict
            dict of test data, output of split_k_fold()
        train : dict
            dict of test data, output of split_k_fold()
        space : dict
            dict of parameters to evaluate. E.g. {'param' : [val1, val2]}
            For iALS models the following parameters are necessary:
                factors, regularization, alpha
        eval : str
            evaluation protocol
            'cv' for k-fold crossvalidation, 'split' for one split
        """

        # prepare parameter space dict
        keys, values = zip(*param_space.items())
        result = [dict(zip(keys, p)) for p in product(*values)]
        
        first_iter = True
        
        #iterate through all param combinations
        for r in result:
            model = implicit.als.AlternatingLeastSquares(factors=r['factors'], regularization=r['regularization'], alpha=r['alpha'])
            
            if eval == 'cv':
                #crossvalidation with method from CrossValidation
                res = self.k_fold_eval(test, train, model, return_type='mean')
            
            if eval == 'split':
                #simple train/test split, applying evaluate_model method from CrossValidation
                res = self.evaluate_model(model, train, test, 10)

            #create final frame in the first iter
            if first_iter == True:
                metrics_frame = res
                first_iter = False
            
            #add metrics of r-th parameter combination to frame
            else:
                metrics_frame = pd.concat((metrics_frame, res), axis=0)
        
        #prepare frame of parameter combinations
        param_df = pd.DataFrame(result)

        #compose frame of parameter combinations and respective metrics 
        ret = pd.concat((param_df.reset_index(drop=True), metrics_frame.reset_index(drop=True)), axis=1)
        return ret
    


