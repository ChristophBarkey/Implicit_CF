# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualize:

    def __init__(self):
        pass


    def get_heatmap(self, result_frame, model_class, metric, save=False):
        if model_class == 'iALS':
            ind = 'alpha'
            col = 'regularization'
            
        if model_class == 'LMF':
            ind = 'learning_rate'
            col = 'regularization'

        if model_class == 'BPR':
            ind = 'learning_rate'
            col = 'regularization'

        heatmap_df = result_frame.pivot(index=ind, columns=col, values=metric)

        if model_class == 'BPR':
            ind = 'learningrate'
            heatmap_df.index.name = ind

        if model_class == 'LMF':
            ind = 'learningrate'
            heatmap_df.index.name = ind

        fig, ax = plt.subplots(figsize=(15, 12), nrows=1, ncols=1)

        plt.style.use('seaborn-whitegrid')
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'cm'
        #sns.set(rc={'text.usetex':True})
        ax = sns.heatmap(heatmap_df, annot=True, cmap='inferno', annot_kws={'fontsize':15}, cbar_kws={'shrink':1.0}, square=False, cbar=True)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        #plt.title('Heatmap', fontsize=30)
        #plt.pcolor(heatmap_df, cmap='inferno')
        ax.set_title(('Heatmap of ' + str(metric) + ' for ' + str(model_class) + ' parameters'), fontsize=30, pad=25)
        ax.set_xlabel(str(col), fontsize=25)
        ax.set_ylabel(str(ind), fontsize=25)
        ax.set_xticks(np.arange(0.5, len(heatmap_df.columns), 1))
        ax.set_yticks(np.arange(0.5, len(heatmap_df.index), 1))
        ax.set_xticklabels(heatmap_df.columns)
        ax.set_yticklabels(heatmap_df.index)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        #plt.colorbar()
        if save:
            plt.savefig('heatmap.pdf')
        plt.show()


