# %%
import pandas as pd
import numpy as np
#from arm_py import AssociationRuleMining, arm_dataprep



def train_test_split_arm(data, split_date):
    train = data[data.create_date <= split_date]
    test = data[data.create_date > split_date]
    test_filtered = filter_test_data(train, test)
    print('Train prop: ' + str(round(len(train) / len(data), 2)))
    return (train, test_filtered)



def filter_test_data(train, test):
    dealers = pd.concat([train, test], axis=0).user.unique()
    test_items_list = []
    test_users_list = []
    for d in dealers:
        train_items = train[train.user == d].item_id.unique()
        test_items = test[test.user == d].item_id.unique()
        test_set = test_items[np.invert(np.isin(test_items, train_items))]
        for i in test_set:
            test_items_list.append(i)
            test_users_list.append(d)
    test_df = pd.DataFrame({'test_user': test_users_list, 'test_items': test_items_list})
    print('Filtered out: ' + str(round(1-(len(test_df) / len(test)), 2)) + ' of test lines')
    return test_df


def filter_min_training_lines(train, test, arm_model):
    num_training_lines_all = train.groupby('item_id')[['user']].count().reset_index()
    min_training_lines = arm_model.min_support * train.co_id.nunique()
    test_users = test.test_user.unique()
    for u in test_users:
        test_subset = test[test.test_user == u]
        training_lines_per_item = pd.merge(test_subset, num_training_lines_all, left_on='test_items', right_on='item_id', how='inner')
        filtered_items = training_lines_per_item[training_lines_per_item.user > min_training_lines]
        if len(filtered_items) > 0:
            if u == test_users[0]:
                ret = filtered_items
            else:
                ret = pd.concat([ret, filtered_items], axis=0)
    return ret


def prepare_recos(results):
    dealers = results.keys()
    dealers_list = []
    recos_list = []
    for d in dealers:
        recos = results[d]
        if len(recos) > 0:
            for i in recos:
                dealers_list.append(d)
                recos_list.append(i)
    recos_df = pd.DataFrame({'user' : dealers_list, 'item' : recos_list})
    #print('Generated recommendations: ' + str(len(recos_df)))
    return recos_df



def precision_per_u(results, model, train, test, return_type='df'):
    recos = prepare_recos(results)
    test_reduced = filter_min_training_lines(train, test, model)
    precision_per_u = []
    user_u = []
    for u in recos.user.unique():
        recommended = recos[recos.user == u].item
        relevent = test_reduced[test_reduced.test_user == u].test_items
        hits = np.isin(recommended, relevent).sum()
        num_recs = len(recommended)
        num_possible_hits = len(relevent)
        if len(relevent) > 0 and len(recommended) > 0:
            precision = hits/min(num_recs, num_possible_hits)
            precision_per_u.append(precision)
            user_u.append(u)
    ret = pd.DataFrame({'user': user_u, 'precision': precision_per_u}).sort_values('precision', ascending=False)
    if return_type == 'df':
        return ret
    if return_type == 'metrics':
        p = ret.precision.mean()
        return {'precision': p, 'num_recs': len(recos)}


