# %%
import pandas as pd
import numpy as np
from implicit.evaluation import train_test_split, ranking_metrics_at_k
import implicit
from eALS_adaptor import eALSAdaptor
from lightFM_adoptor import LightFMAdaptor
from itertools import product


class CrossValidation:
    
    def __init__(self, k):
        """" Crossvalidation

        A class for performing k-fold crossvalidation and hyperparameter tuning. 

        Parameters
        ----------
        user_item : csr_matrix
            Matrix in the format ((user, item), purchases) representing the user item interactions
        k : int
            Number of folds to be performed in crossvalidation
        """
        self.k = k


    def split_k_fold(self, user_item, seed) :
        """" Split k fold

        Function to split the attributed user_item matrix k fold

        Returns
        -------
        (test_dict, train_dict) : (dict of k csr_matrices, dict of k csr_matrices)
            Two dictionaries, containing respectively the train and test data. Each dict contains k csr matrices.
        """
        return_dict = {}
        return_dict_train = {}

        # splitting k-1 times to obtain k sets
        for i in range(self.k-1):

            # dynamically adjust the splitting proportions
            train_temp, test_temp = train_test_split(user_item, train_percentage=((self.k-(i+1))/(self.k-i)), random_state=seed)
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
            user_item = train_temp
        return (return_dict, return_dict_train)


    def hyperp_tuning_simple(self, test, train, seed, param_space, model_class, user_features=None, item_features=None, eval_k=10, no_weights=False,
    exclude=None, mpr=True):
        """" Simplified hyperparameter tuning method for implicit models

        Function to evaluate one model class for a given parameter space. Each model is only evaluatd once on a test set
        Designed for pre-tuning of models

        Parameters
        ----------
        test : csr_matrix
            csr_matrix of test data, output of train_test_split()
        train : csr_matrix
            csr_matrix of train data, output of train_test_split()
        seed : int
            random_state initializer for the model parameter nitialization. 
            For reproducable results.
        param_space : dict
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
            iALS, LMF, BPR, eALS, FM
        return_type : str
            Identifier for the return type of the k-fold eval. Per default 'mean', possible 'full'.
            Should only be changed, if only one parameter combination is applied

        Returns
        -------
        eval_frame : dataframe
            Pandas dataframe containing all evaluated parameter combinations and the respective metric values
        """
        # test and train are csr_matrices, not dicts!!
        # prepare parameter space dict
        keys, values = zip(*param_space.items())

        # result is a list of dicts, each dict is one parameter combination
        result = [dict(zip(keys, p)) for p in product(*values)]
        
        first_iter = True

        #iterate through all param combinations
        for r in result:

            #get model with parameters as indicated and seed
            model = self.get_model(r, model_class, seed)
            
            if model_class == 'FM':
                if no_weights:
                    model.fit(train.sign(), user_features, item_features, show_progress=False)
                else:
                    model.fit(train.sign(), user_features, item_features, train, show_progress=False)

            else:
                try:
                    model.fit(train, show_progress=False)
                
                # if Nan appears in factors, they are transformed to 0 and the param combination printed out
                except:
                    model.user_factors[np.isnan(model.user_factors)] = 0
                    model.item_factors[np.isnan(model.item_factors)] = 0
                    print(r)

            res = self.evaluate_model(model, train, test, eval_k, exclude, mpr)

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


    def hyperp_tuning(self, test, train, exclude, seed, param_space, model_class, return_type='mean', user_features=None, item_features=None, mpr=True):
        """" Hyperparameter tuning method for implicit models

        Function to evaluate one model class for a given parameter space. Each model is then evaluated using k-fold CV

        Parameters
        ----------
        test : dict
            dict of test data, output of split_k_fold()
        train : dict
            dict of test data, output of split_k_fold()
        exclude : csr_matrix
            csr_matrix of interactions that need to be excluded in th evaluation protocol.
            Usually the test set of the initial data split.
            The evaluation function considers still the size of the initial matrix and would generate recommendations also for the initially excluded test set!
        seed : int
            random_state initializer for the model parameter initialization. 
            For reproducable results.
        param_space : dict
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
            iALS, LMF, BPR, eALS, FM
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
            
            #evaluate model on k train/test dicts with k_fold_eval method
            res = self.k_fold_eval(test, train, exclude, r, model_class, seed, return_type=return_type, 
            user_features=user_features, item_features=item_features, mpr=mpr)

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



    # IMPORTANT: here test, train are dicts. Output from split_k_fold()
    def k_fold_eval(self, test, train, exclude, r, model_class, seed, return_type, user_features=None, item_features=None, mpr=True) :
        """" K-fold evaluation

        Function to evaluate one model with given parameter combination k times, applying the k-fold crossvalidation

        Parameters
        ----------
        test : dict of csr_matrices
            Dict containing the k test sets
        train : dict of csr_matrices
            Dict containing the k train sets
        exclude : csr_matrix
            csr_matrix of interactions that need to be excluded in th evaluation protocol.
        r : dict
            Dict containing the parameter combination. E.g. {'param1' : val1, 'param2' : val2}
            Parameters vary according to the model_class
        model_class : str
            Identifier for the model class. 
            iALS, LMF, BPR, eALS
        seed : int
            random_state initializer for the model parameter initialization. 
        return_type : str
            Identifier for the format of the returned frame
            mean, full
        
        Returns
        -------
        eval_frame : dataframe
            Pandas dataframe containing the mean of the k folds.
            If return_type=mean, the k folds are averaged, if return_type=full, each fold is returned
        """

        # going through all k folds
        # for i in range(self.k) : cleaner?
        for i in range(len(test)) :

            # get an empty model according to model_class and parameter combination and seed
            model = self.get_model(r, model_class, seed)
            
            # pick the i-th train and test matrix from the dicts
            test_temp = test[str(i)]
            train_temp = train[str(i)]

            # in case of FM model, fit method needs additional features
            if model_class == 'FM':
                model.fit(train_temp.sign(), user_features, item_features, train_temp, show_progress=False)

            else:
                # for the BPR model sometimes NaNs appear in the factors and an error interrupts the tuning
                try:
                    model.fit(train_temp, show_progress=False)
                
                # if Nan appears in factors, they are transformed to 0 and the param combination printed out
                except:
                    model.user_factors[np.isnan(model.user_factors)] = 0
                    model.item_factors[np.isnan(model.item_factors)] = 0
                    print(r)

            # after fitting the model, it is evaluated. Using k=10 as default for ranking_matrics_at_k
            m = self.evaluate_model(model, train_temp, test_temp, 10, exclude, mpr)
            if i == 0:
                df = m
            else :
                df = pd.concat((df, m), axis=0)
        if return_type == 'full':
            return df
        if return_type == 'mean':
            return df.mean().to_frame().T

    
    def evaluate_model(self, model, train, test, k, exclude=None, mpr=True):
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
        exclude : csr_matrix
            csr_matrix of interactions that need to be excluded in th evaluation protocol.
        k : int
            Number of top recommendations to be evaluated

        Returns
        -------
        metrics : dataframe
            Pandas dataframe containing all metrics
        """
        # exclude should be defined for the k-fold cv case, in the simplified version not necessary
        if exclude is not None:

            # observations from train AND the initial test set (exclude) should not be used for evaluation!
            disregard_obs = (train + exclude)
        else:
            # still exclude train observations in the simplified case
            disregard_obs = train

        # evaluating the model on the validation 
        metrics = ranking_metrics_at_k(model, disregard_obs, test, K=k, show_progress=False)
       
        # mpr computation can be switched off
        if mpr:
            mpr = self.mpr_new(model, disregard_obs, test)
            metrics.update(mpr)
        return pd.DataFrame(metrics, index=['metrics@'+str(k)])  

    
    # new MPR function applying the exact formula, considering the real rt values
    # Fast for smaller matrices, crashes for large matrices!
    def mpr_new(self, model, disregard_obs, test):
        """" MPR calculation function

        Calculates the MPR over all users, disregarding the training items and initial test items.
        Oriented on the original introduction by Hu. et al. 2008. considering the real observation values.
        Faster than the per_user version, but crashes for large matrices.

        Parameters
        ----------
        model : implicit model
            Fitted implicit model
        disregard_obs : csr_matrix
            csr_matrix of interactions that need to be excluded in th evaluation protocol.
            Union of train and initial test (exclude)
            If the method is used individually, plug in train set for disregard_obs
        test : csr_matrix
            Matrix containing interactions that were held out for training

        Returns
        -------
        mpr : dict
            Dictionary containing the MPR
        """
        num_users = test.shape[0]
        num_items = test.shape[1]

        # get recommendations for all users and all items, disregarding the train + exclude obs
        rec_items = model.recommend(np.arange(num_users), disregard_obs, num_items)[0]

        # generate rank matrix with equal dimensions
        rank = np.array([np.arange(num_items) / (num_items-1)])
        rank_rep = np.tile(rank, num_users).reshape(num_users, num_items)

        # sort rank matrix according to recommendations
        sort_rank = rank_rep[np.arange(num_users)[:, None], rec_items.argsort()]
        
        # multiply sorted rank matrix with real test observations, zeros are disregarded
        rt_rank = test.toarray() * sort_rank

        # divide sum of observations * rank by the sum of observations
        mpr = rt_rank.sum() / test.toarray().sum()

        return {'mpr' : mpr} 


    # new MPR function applying the exact formula, considering the real rt values
    # Works for larger matrices, but a bit slower
    # Not necessary for existing OEM data, therefore not used
    def MPR_new_per_user(self, model, train, test):
        num_users = test.shape[0]
        num_items = test.shape[1]

        rank = np.arange(num_items) / (num_items-1)

        rt_rank = 0
        rt = 0

        for u in range(num_users):
            # get recommendations for all users and all items
            rec_items = model.recommend(u, train[u], num_items)[0]

            # sort rank matrix according to recommendations
            sort_rank = rank[rec_items.argsort()]

            # multiply sorted rank matrix with real test observations, zeros are disregarded
            rt_rank += (test[u].toarray() * sort_rank).sum()

            rt += test[u].toarray().sum()

        return {'mpr' : rt_rank / rt}  
    
   
    def evaluate_at_k_new(self, param_space, model_class, train, test, max_k, user_features=None, item_features=None):
        """" Evaluate fitted model on different values of k

        Method for final model comparison.
        Model is fitted on optimal param combination and then evaluated repeatedly applying different k values.

        Parameters
        ----------
        param_space : dict
            dict of parameters to evaluate. 
            In this case, only ONE parameter combination should be passed
        model_class : str
            Identifier for the model class
        test : csr_matrix
            csr_matrix of test data, output of train_test_split()
        train : csr_matrix
            csr_matrix of train data, output of train_test_split()
        max_k : int
            maximum number of k to be evaluated.
            All positive integer values k < max_k will be evaluated
        Returns
        -------
        res_df : Dataframe
            Pandas dataframe containing evlauation values for all k values
        """    
        keys, values = zip(*param_space.items())

        # result is a list of dicts, each dict is one parameter combination
        result = [dict(zip(keys, p)) for p in product(*values)]

        #iterate through all param combinations
        r = result[0]

        model = self.get_model(r, model_class, seed=22)

        if model_class == 'FM':
            model.fit(train.sign(), user_features, item_features, train, show_progress=False)

        else:
            try:
                model.fit(train, show_progress=False)
            
            # if Nan appears in factors, they are transformed to 0 and the param combination printed out
            except:
                model.user_factors[np.isnan(model.user_factors)] = 0
                model.item_factors[np.isnan(model.item_factors)] = 0
                print(r)

        for k in range(max_k):
            eval_df = self.evaluate_model(model, train, test, k+1)
            eval_df.index = [k+1]
            if k == 0:
                res_df = eval_df
            else:
                res_df = pd.concat([res_df, eval_df], axis=0)       
        return res_df 


    # Function to evaluate FM model on different set of user and/or item features
    # Invokes hyperp_simple for actual tuning, once the features are defined
    def tune_FM(self, space, user_f, item_f, uf_names, if_names, train, test, exclude):
        results = []
        for u in range(len(user_f)):
            for i in range(len(item_f)):
                res = self.hyperp_tuning_simple(test=test, train=train, seed=22, param_space=space, model_class='FM', user_features=user_f[u], 
                item_features=item_f[i], exclude=exclude)
                res['uf_name'] = uf_names[u]
                res['if_name'] = if_names[i]
                results.append(res)
        
        # prep return frame
        for i in range(len(results)):
            if i == 0:
                ret_df = results[i].copy()
            else:
                ret_df = pd.concat([ret_df, results[i].copy()])
        
        return ret_df


    def get_model(self, p, model_class, seed):
        """"Method to get model according to class and params
        
        Parameters
        ----------
        p : dict
            Dictionary containing one parameter combination. Params vary according to model_class
        model_class : str
            specifying the model class, iALS, LMF, BPR, eALS
        seed : int
            random seed for parameter initialization

        Returns
        -------
        model : implicit/eals model
            empty model, with the given param combination
        """
        if model_class == 'iALS':
            model = implicit.als.AlternatingLeastSquares(factors=p['factors'], regularization=p['regularization'], 
            alpha=p['alpha'], iterations=p['iterations'], num_threads=4, random_state=seed)
        
        if model_class == 'LMF':
            model = implicit.lmf.LogisticMatrixFactorization(factors=p['factors'], learning_rate=p['learning_rate'], 
            regularization=p['regularization'], iterations=p['iterations'], neg_prop=p['neg_prop'], random_state=seed)
        
        if model_class == 'BPR':
            model = implicit.bpr.BayesianPersonalizedRanking(factors=p['factors'], learning_rate=p['learning_rate'], 
            regularization=p['regularization'], iterations=p['iterations'], random_state=seed)

        if model_class == 'eALS':
            model = eALSAdaptor(factors=p['factors'], alpha=p['alpha'], 
            regularization=p['regularization'], w0=p['w0'], num_iter=p['iterations'], random_state=seed)

        if model_class == 'FM':
            model = LightFMAdaptor(no_components=p['factors'], learning_rate=p['learning_rate'], loss=p['loss'], item_alpha=p['regularization'], 
            user_alpha=p['regularization'], max_sampled=p['max_sampled'], iterations=p['iterations'], random_state=seed)
        
        return model



# %%
