# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualize:

    def __init__(self):
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'cm'


    def get_heatmap(self, result_frame, model_class, save=False):
        if model_class == 'iALS':
            ind = 'alpha'
            col = 'regularization'
            
        if model_class == 'LMF':
            ind = 'learning_rate'
            col = 'regularization'

        if model_class == 'BPR':
            ind = 'learning_rate'
            col = 'regularization'

        if model_class == 'eALS':
            ind = 'w0'
            col = 'regularization'

        df_1 = result_frame.pivot(index=ind, columns=col, values='precision')
        df_2 = result_frame.pivot(index=ind, columns=col, values='map')
        df_3 = result_frame.pivot(index=ind, columns=col, values='ndcg')
        df_4 = result_frame.pivot(index=ind, columns=col, values='mpr')

        data = [df_1, df_2, df_3, df_4]
        titles = ['P@10 for ' + str(model_class) + ' parameters', 
                    'MAP@10 for ' + str(model_class) + ' parameters',
                    'NDCG@10 for ' + str(model_class) + ' parameters',
                    'MPR for ' + str(model_class) + ' parameters']

        if model_class == 'iALS':
            ind = r'alpha $\alpha$'
            col = r'regularization $\lambda$'
        
        if model_class == 'BPR':
            ind = 'learningrate'
            heatmap_df.index.name = ind

        if model_class == 'LMF':
            ind = 'learningrate'
            heatmap_df.index.name = ind

        fig, ax = plt.subplots(figsize=(20, 20), nrows=2, ncols=2, sharex=True, sharey=True)

        plt.subplots_adjust(wspace=0.1)

        c = 0
        for i in range(2):
            for j in range(2):

                heatmap_df = data[c]

                if c == 3:
                    sns.heatmap(heatmap_df, annot=True, cmap='inferno_r', annot_kws={'fontsize':15}, cbar_kws={'shrink':1.0}, square=False, cbar=False, ax=ax[i, j])
                else:
                    sns.heatmap(heatmap_df, annot=True, cmap='inferno', annot_kws={'fontsize':15}, cbar_kws={'shrink':1.0}, square=False, cbar=False, ax=ax[i, j])
                #cbar = ax.collections[0].colorbar
                #cbar.ax.tick_params(labelsize=15)
                #plt.title('Heatmap', fontsize=30)
                #plt.pcolor(heatmap_df, cmap='inferno')
                ax[i, j].set_title((titles[c]), fontsize=30, pad=25)
                ax[i, j].set_xlabel(str(col), fontsize=25)
                ax[i, j].set_ylabel(str(ind), fontsize=25)
                ax[i, j].set_xticks(np.arange(0.5, len(heatmap_df.columns), 1))
                ax[i, j].set_yticks(np.arange(0.5, len(heatmap_df.index), 1))
                ax[i, j].set_xticklabels(heatmap_df.columns)
                ax[i, j].set_yticklabels(heatmap_df.index)
                ax[i, j].tick_params(axis='both', which='major', labelsize=15)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=15)

                c += 1

        #sns.set(rc={'text.usetex':True})

        #plt.colorbar()
        if save:
            plt.savefig('heatmap.pdf', bbox_inches='tight')

        plt.show()


