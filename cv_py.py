# %%
import pandas as pd
import numpy as np
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from implicit.datasets.movielens import get_movielens
import implicit


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
        metrics = ranking_metrics_at_k(model, train, test, K=k)
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


    def k_fold_eval(self, test, train, model, alpha) :
        for i in range(len(test)) :
            model = model
            test_temp = test[str(i)]
            train_temp = train[str(i)]
            print(test_temp.nnz)
            print(train_temp.nnz)
            model.fit(train_temp * alpha)
            m = self.evaluate_model(model, train_temp, test_temp, 10)
            if i == 0:
                df = m
            else :
                df = pd.concat((df, m), axis=0)
        return df



# %%
