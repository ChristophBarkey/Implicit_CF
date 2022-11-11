# %%
# %%
import pandas as pd
import numpy as np
import implicit
from implicit.evaluation import ranking_metrics_at_k, train_test_split
from implicit.nearest_neighbours import CosineRecommender
from implicit.als import AlternatingLeastSquares
from eALS_adaptor import eALSAdaptor
from implicit.lmf import LogisticMatrixFactorization
from implicit.bpr import BayesianPersonalizedRanking
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

class Recommender:

    def __init__(self):
        pass

    def get_recommendations(self, model, user_item_csr, user_item_full):
        recoms_t = model.recommend(userid=pd.Series(user_item_csr.tocoo().row).unique(), user_items=user_item_csr)

        for u in range(len(recoms_t[0])):
            recos_u = recoms_t[0][u]
            items_u = user_item_full[user_item_full.item_codes.isin(recos_u)]['item'].unique()
            user = user_item_full[user_item_full.user_codes==u]['user'].unique()[0]
            df = pd.DataFrame({user : items_u}).T
            if u == 0:
                ret_df = df
            else:
                ret_df = pd.concat([ret_df, df], axis=0)
        return ret_df
    
    def get_common_recos_t(self, recos_ials_t, recos_eals_t, recos_cosine_t):
        all_three = []
        ials_eals = []
        ials_cosi = []
        eals_cosi = []

        for u in range(len(recos_ials_t)):
            ials = set(list(recos_ials_t.iloc[u, :]))
            eals = set(list(recos_eals_t.iloc[u, :]))
            cosi = set(list(recos_cosine_t.iloc[u, :]))

            all_three.append(len(ials & eals & cosi))
            ials_eals.append(len(ials & eals))
            ials_cosi.append(len(ials & cosi))
            eals_cosi.append(len(eals & cosi))

        common_recos_df = pd.DataFrame({'all three' : all_three, 'ials & eals' : ials_eals, 'ials & cosi' : ials_cosi, 'eals & cosi' : eals_cosi}, index=recos_ials_t.index)
        return common_recos_df

    def get_common_recos_a(self, recos_ials_a, recos_bpr_a, recos_lmf_a, recos_cosine_a):
        all_four = []

        ials_bpr_lmf = []
        ials_bpr_cosi = []
        ials_lmf_cosi = []
        bpr_lmf_cosi = []

        ials_bpr = []
        ials_lmf = []
        ials_cosi = []
        bpr_lmf = []
        bpr_cosi = []
        lmf_cosi = []

        for u in range(len(recos_ials_a)):
            ials = set(list(recos_ials_a.iloc[u, :]))
            bpr = set(list(recos_bpr_a.iloc[u, :]))
            lmf = set(list(recos_lmf_a.iloc[u, :]))
            cosi = set(list(recos_cosine_a.iloc[u, :]))

            all_four.append(len(ials & bpr & lmf & cosi))

            ials_bpr_lmf.append(len(ials & bpr & lmf))
            ials_bpr_cosi.append(len(ials & bpr & cosi))
            ials_lmf_cosi.append(len(ials & lmf & cosi))
            bpr_lmf_cosi.append(len(bpr & lmf & cosi))

            ials_bpr.append(len(ials & bpr))
            ials_lmf.append(len(ials & lmf))
            ials_cosi.append(len(ials & cosi))
            bpr_lmf.append(len(bpr & lmf))
            bpr_cosi.append(len(bpr & cosi))
            lmf_cosi.append(len(lmf & cosi))

        common_recos_df = pd.DataFrame({'all_four' : all_four, 'ials_bpr_lmf' : ials_bpr_lmf, 'ials_bpr_cosi' : ials_bpr_cosi, 'ials_lmf_cosi' : ials_lmf_cosi,
        'bpr_lmf_cosi' : bpr_lmf_cosi, 'ials_bpr' : ials_bpr, 'ials_lmf' : ials_lmf, 'ials_cosi' : ials_cosi, 'bpr_lmf' : bpr_lmf, 'bpr_cosi' : bpr_cosi,
        'lmf_cosi' : lmf_cosi}, index=recos_ials_a.index)
        return common_recos_df




