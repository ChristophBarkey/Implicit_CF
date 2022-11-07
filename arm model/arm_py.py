# %%
import pandas as pd
import numpy as np
from apyori import apriori
from arm_evaluation_py import precision_per_u

def arm_data_import(OEM):
    if OEM == 'AGCO':
        agco_cod_18 = pd.read_csv('cod_agco_new_2018.csv', sep = '|', low_memory=False)
        agco_cod_19 = pd.read_csv('cod_agco_new_2019.csv', sep = '|', low_memory=False)
        agco_cod_20 = pd.read_csv('cod_agco_new_2020.csv', sep = '|', low_memory=False)
        agco_cod_21 = pd.read_csv('cod_agco_new_2021.csv', sep = '|', low_memory=False)

        transactions_t = pd.concat([agco_cod_18, agco_cod_19, agco_cod_20, agco_cod_21])
        locations_t = pd.read_csv('loc_agco_new.csv', sep = '|', low_memory=False)

    if OEM == 'TEREX':
        transactions_t = pd.read_csv('cod_terex_new.csv', sep='|', low_memory=False)
        locations_t = pd.read_csv('loc_terex_new.csv', sep = '|', low_memory=False)

    orders_filtered = pd.merge(transactions_t, locations_t, left_on='supply_location_id', right_on='location_id', how='inner')

    ret = orders_filtered.copy()
    duplicate_co_id = ret.co_id.value_counts() > 1
    ret_filtered = ret[ret.co_id.isin(duplicate_co_id[duplicate_co_id].index)]

    return ret_filtered

class AssociationRuleMining:

    def __init__(self, min_support=0.1, min_confidence=0.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = None

    def fit(self, train):
        transaction_list = list(train.groupby('co_id')['item_id'].apply(list))
        results = list(apriori(transactions=transaction_list, min_support=self.min_support, min_confidence=self.min_confidence))
        rules = self._get_rules_df(results)
        self.rules = rules.copy()

    def tune_arm(self, train, test, support_list, confidence_list):
        self.min_support = min(support_list)
        self.min_confidence = min(confidence_list)

        self.fit(train)

        rules_temp = self.rules.copy()

        first_iter = True
        for s in support_list:
            rules_s = rules_temp[rules_temp.supp >= s].copy()
            for c in confidence_list:
                rules_s_c = rules_s[rules_s.conf >= c].copy()

                results_temp = self.get_candidates(train, rules_s_c)

                metrics_temp = precision_per_u(results_temp, s, train, test, 'metrics')

                params = {'supp': s, 'conf': c}
                params.update(metrics_temp)

                df = pd.DataFrame(params, index=[0])

                if first_iter:
                    ret = df
                    first_iter = False
                else:
                    ret = pd.concat([ret, df], axis=0)
        
        return ret


    def get_candidates(self, train, rules=None):

        if rules is None:
            rules = self.rules

        cod = train[['user', 'item_id']].drop_duplicates()


        # cod containing user column and item column
        test_base = self._unfreeze(rules.base).reset_index(drop=True)
        test_add = self._unfreeze(rules['add']).reset_index(drop=True)

        return_dict = {}
        for u in cod.user.unique():
            items_u = cod[cod.user == u][['item_id']].drop_duplicates()

            inside_mask = self._isin_base(test_base.unfrozen, items_u.item_id)
            if len(inside_mask) == 0:
                continue
            
            df_inside = test_add[inside_mask].reset_index(drop=True)

            df_inside_unlisted = self._unlist_df(df_inside.unfrozen)

            df_outside = df_inside_unlisted[np.invert(df_inside_unlisted.unlisted.isin(items_u.item_id))].drop_duplicates()

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



