# %%
import pandas as pd
import numpy as np
from apyori import apriori

class AssociationRuleMining:

    def __init__(self, min_support=0.1, min_confidence=0.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = None

    def fit(self, transaction_list):
        results = list(apriori(transactions=transaction_list, min_support=self.min_support, min_confidence=self.min_confidence))
        rules = self._get_rules_df(results)
        self.rules = rules


    def get_candidates(self, cod):

        # cod containing user column and item column
        test_base = self._unfreeze(self.rules.base).reset_index(drop=True)
        test_add = self._unfreeze(self.rules['add']).reset_index(drop=True)

        return_dict = {}
        for u in cod.user.unique():
            items_u = cod[cod.user == u][['item_id']].drop_duplicates()

            inside_mask = self._isin_base(test_base.unfrozen, items_u.item_id)
            df_inside = test_add[inside_mask].reset_index(drop=True)

            df_inside_unlisted = self._unlist_df(df_inside.unfrozen)

            df_outside = df_inside_unlisted[np.invert(df_inside_unlisted.unlisted.isin(items_u.item_id))]

            return_dict.update({u : list(df_outside.unlisted.values)})

        return return_dict

    def _unfreeze(self, frozen):
        unfrozen_1 = []
        for i in frozen:
            unfrozen_1.append(list(i))
        df1 = pd.DataFrame({'unfrozen' : unfrozen_1})
        return df1

    def _get_rules_df(self, results):
        supports = []
        bases = []
        adds = []
        confidences = []
        lifts = []

        for i in range(len(results)):
            rule_items = results[i][0]
            support = results[i][1]
            rules = results[i][2]
            for r in range(len(rules)):
                rule = results[i][2][r]
                base = rule[0]
                add = rule[1]
                confidence = rule[2]
                lift = rule[3]
                
                supports.append(support)
                bases.append(base)
                adds.append(add)
                confidences.append(confidence)
                lifts.append(lift)


        return pd.DataFrame({'base' : bases, 'add' : adds, 'conf' : confidences, 'supp' : supports, 'lift' : lifts})
         

    def _isin_base(self, base, items):
        mask = []
        for i in range(len(base)):
            boolean = set(base[i]).issubset(items)
            mask.append(boolean)
        return mask

    def _unlist_df(self, df_listed):
        unlisted = []
        for i in range(len(df_listed)):
            for j in range(len(df_listed[i])):
                unlisted.append(df_listed[i][j])
        return pd.DataFrame({'unlisted' : unlisted})


