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
        """" Crossvalidation

        A class for performing k-fold crossvalidation and hyperparameter tuning. 

        Parameters
        ----------
        user_item : csr_matrix
            Matrix in the format ((user, item), purchases) representing the user item interactions
        k : int
            Number of folds to be performed in crossvalidation
        """
        self.user_item = user_item
        self.k = k

    def mpr_per_user(self, model, train, test, num_recs, user):
        """" MPR per user

        Calculates the MPR for given user

        Parameters
        ----------
        model : implicit model
            Fitted implicit model
        train : csr_matrix
            Matrix used for model training
        test : csr_matrix
            Matrix containing interactions that were held out for training
        num_recs : int
            Number of recommendations that the MPR calculation will be based on
        user : int
            User id of the user the MPR is calculated for

        Returns
        -------
        MPR : float
            Mean percentage ranking for the given user
        """
        
        # generate ids of recommendation list for given user of length num_recs
        recommended_items = model.recommend(user_items=train[user], userid=user, filter_already_liked_items=True, N = num_recs)[0]
        
        # get list of test items for the given user 
        test_items = test[user].nonzero()[1]

        # filter test items for hits by the recommendation list
        test_items_in_list = test_items[np.isin(test_items, recommended_items)]
        
        # if the number of hits is 0, the default value 0.5 is returned, which is expected for a random recommender
        if len(test_items_in_list) == 0:
            return 0.5

        recommended_indices = recommended_items.argsort()
        hit_indices = recommended_indices[np.searchsorted(recommended_items[recommended_indices], test_items_in_list)]
        #return (np.sum(hit_indices) / num_recs) / len(hit_indices)
        return np.mean(hit_indices / num_recs)
   
    def calc_mpr(self, model, train, test):
        """" MPR overall

        Calculates the MPR over a defined set of users

        Parameters
        ----------
        model : implicit model
            Fitted implicit model
        train : csr_matrix
            Matrix used for model training
        test : csr_matrix
            Matrix containing interactions that were held out for training
        
        Returns
        -------
        MPR : dict
            MPR over all users
        """
        mprs = []

        # going through all users of the user_item matrix
        for u in range(self.user_item.shape[0]) :

            # calculate the MPR for the given user, with the number of all items as length of recommendation list
            mpr = self.mpr_per_user(model, train, test, self.user_item.shape[1], u)
            mprs.append(mpr)
        return {'mpr' : np.mean(mprs)} 
   

    def evaluate_model(self, model, train, test, k):
        """" Evaluation Function

        Wrapper function including the ranking_at_k_metrics and the MPR metric, returning one frame with all metrics

        Parameters
        ----------
        model : implicit model
            Fitted implicit model
        train : csr_matrix
            Matrix used for model training
        test : csr_matrix
            Matrix containing interactions that were held out for training
        k : int
            Number of top recommendations to be evaluated

        Returns
        -------
        metrics : dataframe
            Pandas dataframe containing all metrics
        """
        metrics = ranking_metrics_at_k(model, train, test, K=k, show_progress=False)
        mpr = self.calc_mpr(model, train, test)
        metrics.update(mpr)
        return pd.DataFrame(metrics, index=['metrics@'+str(k)])  
   
    def split_k_fold(self) :
        """" Split k fold

        Function to split the attributed user_item matrix k fold

        Returns
        -------
        (test_dict, train_dict) : (dict of k csr_matrices, dict of k csr_matrices)
            Two dictionaries, containing respectively the train and test data. Each dict contains k csr matrices.
        """
        split_matrix = self.user_item
        return_dict = {}
        return_dict_train = {}

        # splitting k-1 times to obtain k sets
        for i in range(self.k-1):

            # dynamically adjust the splitting proportions
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
    def k_fold_eval(self, test, train, r, model_class, return_type) :
        """" K-fold evaluation

        Function to evaluate one model with given parameter combination k times, applying the k-fold crossvalidation

        Parameters
        ----------
        test : dict of csr_matrices
            Dict containing the k test sets
        train : dict of csr_matrices
            Dict containing the k train sets
        r : dict
            Dict containing the parameter combination. E.g. {'param1' : val1, 'param2' : val2}
            Parameters vary according to the model_class
        model_class : str
            Identifier for the model class. 
            iALS, LMF, BPR, eALS
        return_type : str
            Identifier for the format of the returned frame
            mean, full
        
        Returns
        -------
        eval_frame : dataframe
            Pandas dataframe containing the mean of the k folds.
            If return_type=mean, the k folds are averaged, if return_type=full, each fold is returned
        """

        # going through all k iterations
        for i in range(len(test)) :

            # get an empty model according to model_class and parameter combination
            model = self.get_model(r, model_class)
            
            # pick the i-th train and test matrix from the dicts
            test_temp = test[str(i)]
            train_temp = train[str(i)]

            print(r)

            # eALS is from a different library, hence the additional transformation
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
                try:
                    model.fit(train_temp, show_progress=False)
                
                # if Nan appears in factors, they are transformed to 0
                except:
                    model.user_factors[np.isnan(model.user_factors)] = 0
                    model.item_factors[np.isnan(model.item_factors)] = 0
                    print(r)

            # after fitting the model, it is evaluated. Using k=10 as default for ranking_matrics_at_k
            m = self.evaluate_model(model, train_temp, test_temp, 10)
            if i == 0:
                df = m
            else :
                df = pd.concat((df, m), axis=0)
        if return_type == 'full':
            return df
        if return_type == 'mean':
            return df.mean().to_frame().T

    def hyperp_tuning(self, test, train, param_space, model_class, return_type='mean'):
        """" Hyperparameter tuning method for implicit models

        Function to evaluate one model class for a given parameter space. Each model is then evaluated using k-fold CV

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
            Parameters for eALS:
                factors, alpha, regularization, w0
        model_class : str
            iALS, LMF, BPR, eALS
        return_type : str
            Identifier for the return type of the k-fold eval. Per default 'mean', possible 'full'.
            Should only be changed, if only one parameter combination is applied

        Returns
        -------
        eval_frame : dataframe
            Pandas dataframe containing all evaluated parameter combinations and the respective metric values
        """

        # prepare parameter space dict
        keys, values = zip(*param_space.items())

        # result is a list of dicts, each dict is one parameter combination
        result = [dict(zip(keys, p)) for p in product(*values)]
        
        first_iter = True
        
        #iterate through all param combinations
        for r in result:

            #get model with parameters as indicated
            #model = self.get_model(r, model_class)
            
            #evaluate model on train/test with k_fold_eval
            res = self.k_fold_eval(test, train, r, model_class, return_type=return_type)

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
        p : dict
            Dictionary containing one parameter combination. Params vary according to model_class
        model_class : str
            specifying the model class, iALS, LMF, BPR, eALS

        Returns
        -------
        model : implicit/eals model
            empty model, with the given param combination
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
