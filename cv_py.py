# %%
import pandas as pd
import numpy as np
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from implicit.datasets.movielens import get_movielens
import implicit
from eals import ElementwiseAlternatingLeastSquares, load_model
from itertools import product


class CrossValidation:
    
    def __init__(self, user_item, k):
        self.user_item = user_item
        self.k = k

    def mpr_per_user(self, model, train, test, num_recs, user):
        recommended_items = model.recommend(user_items=train[user], userid=user, filter_already_liked_items=True, N = num_recs)[0]
        test_items = test[user].nonzero()[1]
        test_items_in_list = test_items[np.isin(test_items, recommended_items)]
        if len(test_items_in_list) == 0:
            return 0.5
        recommended_indices = recommended_items.argsort()
        hit_indices = recommended_indices[np.searchsorted(recommended_items[recommended_indices], test_items_in_list)]
        #return (np.sum(hit_indices) / num_recs) / len(hit_indices)
        return np.mean(hit_indices / num_recs)
   
    def calc_mpr(self, model, train, test):
        mprs = []
        for u in range(self.user_item.shape[0]) :
            mpr = self.mpr_per_user(model, train, test, self.user_item.shape[1], u)
            mprs.append(mpr)
        return {'mpr' : np.mean(mprs)} 
   
    def evaluate_model(self, model, train, test, k):
        metrics = ranking_metrics_at_k(model, train, test, K=k, show_progress=False)
        mpr = self.calc_mpr(model, train, test)
        metrics.update(mpr)
        return pd.DataFrame(metrics, index=['metrics@'+str(k)])  
   
    def split_k_fold(self) :
        split_matrix = self.user_item
        return_dict = {}
        return_dict_train = {}
        for i in range(self.k-1):
            train_temp, test_temp = train_test_split(split_matrix, train_percentage=((self.k-(i+1))/(self.k-i)))
            return_dict[str(i)] = test_temp
            if i == 0:
                return_dict_train[str(i)] = train_temp
                rest = test_temp
            else:
                return_dict_train[str(i)] = (train_temp + rest)
                rest = (rest + test_temp)
            if i == (self.k-2):
                return_dict[str(i+1)] = train_temp
                return_dict_train[str(i+1)] = rest
            split_matrix = train_temp
        return (return_dict, return_dict_train)


        # WICHTIG: hier test, train sind dicts. Output von split_k_fold()
    def k_fold_eval(self, test, train, model, model_class, return_type) :
        for i in range(len(test)) :
            model = model
            test_temp = test[str(i)]
            train_temp = train[str(i)]
            #print(test_temp.nnz)
            #print(train_temp.nnz)
            if model_class == 'eALS':
                model.fit(train_temp)
                #create empty implicit model
                temp_model = implicit.als.AlternatingLeastSquares()
                #copy factors from fitted eals model to empty implicit model
                temp_model.user_factors = model.user_factors
                temp_model.item_factors = model.item_factors
                #continue with implicit model to enable evaluation methods
                model = temp_model
            else:
                model.fit(train_temp, show_progress=False)
            m = self.evaluate_model(model, train_temp, test_temp, 10)
            if i == 0:
                df = m
            else :
                df = pd.concat((df, m), axis=0)
        if return_type == 'full':
            return df
        if return_type == 'mean':
            return df.mean().to_frame().T

    def hyperp_tuning(self, test, train, param_space, model_class):
        """" Hyperparameter tuning method for implicit models

        Parameters
        ----------
        test : dict
            dict of test data, output of split_k_fold()
        train : dict
            dict of test data, output of split_k_fold()
        space : dict
            dict of parameters to evaluate. E.g. {'param' : [val1, val2]}
            Parameters for iALS:
                factors, regularization, alpha, iterations
            Parameters for LMF:
                factors, learning_rate, regularization, iterations, neg_prop
            Parameters for BPR:
                factors, learning_rate, regularization, iterations
        model_class : str
            iALS, LMF or BPR
        """

        # prepare parameter space dict
        keys, values = zip(*param_space.items())
        result = [dict(zip(keys, p)) for p in product(*values)]
        
        first_iter = True
        
        #iterate through all param combinations
        for r in result:

            #get model with parameters as indicated
            model = self.get_model(r, model_class)
            
            #evaluate model on train/test with k_fold_eval
            res = self.k_fold_eval(test, train, model, model_class, return_type='mean')

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


    def get_model(self, p, model_class):
        """"Method to get model according to class and params
        
        Parameters
        ----------
        p : list of dicts
            each dict represents a param combination
        model_class : str
            specifying the model class, iALS, LMF or BPR
        """
        if model_class == 'iALS':
            model = implicit.als.AlternatingLeastSquares(factors=p['factors'], regularization=p['regularization'], 
            alpha=p['alpha'], iterations=p['iterations'], num_threads=4)
        
        if model_class == 'LMF':
            model = implicit.lmf.LogisticMatrixFactorization(factors=p['factors'], learning_rate=p['learning_rate'], 
            regularization=p['regularization'], iterations=p['iterations'], neg_prop=p['neg_prop'])
        
        if model_class == 'BPR':
            model = implicit.bpr.BayesianPersonalizedRanking(factors=p['factors'], learning_rate=p['learning_rate'], 
            regularization=p['regularization'], iterations=p['iterations'])

        if model_class == 'eALS':
            model = ElementwiseAlternatingLeastSquares(factors=p['factors'], alpha=p['alpha'], 
            regularization=p['regularization'], w0=p['w0'])
        
        return model



# %%
