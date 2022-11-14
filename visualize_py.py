# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualize:

    # Class containing function to plot specific results
    def __init__(self, style):
        if style == 'seaborn':
            plt.style.use('seaborn-whitegrid')
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = 'cm'
        if style == 'science':
            plt.style.use(['science', 'ieee', 'high-vis'])


    # Function to plot heatmap of results of hyperparameter tuning
    # NOT USED IN THESIS, NOT DEVELOPED COMPLETELY
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

    # Function to plot the tuning results regarding factors and iterations
    def get_convergence_curves(self, result_frame, save=False):
    
        # data preparation
        p_df = result_frame.pivot(index='iterations', columns='factors', values='precision')
        map_df = result_frame.pivot(index='iterations', columns='factors', values='map')
        ndcg_df = result_frame.pivot(index='iterations', columns='factors', values='ndcg')
        mpr_df = result_frame.pivot(index='iterations', columns='factors', values='mpr')
        data_df = [p_df, map_df, ndcg_df, mpr_df]
        names = ['P@10', 'MAP@10', 'NDCG@10', 'MPR']

        # plotting of the four curves
        fig, ax = plt.subplots(figsize=(17, 15), nrows=2, ncols=2)
        plt.subplots_adjust(wspace=0.2, hspace=0.3, right=0.82)
        c = 0
        for i in range(2):
            for j in range(2):
                data_filtered = data_df[c].loc[4:, [10, 50, 100, 150, 200]]
                ax[i,j].plot(data_filtered, linestyle='-', marker='o', linewidth=2)
                ax[i,j].set_title(names[c], fontsize=30)
                ax[i,j].set_xlabel('Iterations', fontsize=25)
                ax[i,j].tick_params(axis='both', which='major', labelsize=17)
                ax[i,j].tick_params(axis='both', which='minor', labelsize=17)
                ax[i,j].autoscale()
                ax[i,j].grid(linestyle=':')
                c += 1
        fig.legend(data_filtered.columns, loc='center right', ncol=1, title='Factors',fancybox=True, shadow=False, frameon=True, title_fontsize=25, fontsize=20,
        bbox_to_anchor=(0.94, 0.79))
        if save:
            plt.savefig('curves.pdf')
        plt.show()


    # Function to plot final model comparison
    def get_comparison_curves(self, precision_df, map_df, ndcg_df, save=False):

        data_df = [precision_df, map_df, ndcg_df]
        names = ['P@k', 'MAP@k', 'NDCG@k']

        # plotting of the three curves
        fig, ax = plt.subplots(figsize=(26, 8), nrows=1, ncols=3)
        plt.subplots_adjust(wspace=0.15, hspace=0.3, right=0.82, bottom=0.25)
        c = 0
        for i in range(3):
            data_filtered = data_df[c]
            ax[i].plot(data_filtered, linestyle='-', marker='o', linewidth=2)
            ax[i].set_title(names[c], fontsize=30)
            ax[i].set_xlabel('k', fontsize=25)
            ax[i].tick_params(axis='both', which='major', labelsize=17)
            ax[i].tick_params(axis='both', which='minor', labelsize=17)
            ax[i].autoscale()
            ax[i].grid(linestyle=':')
            c += 1
        fig.legend(data_filtered.columns, loc='lower center', ncol=4, title='Model',fancybox=True, shadow=False, frameon=True, title_fontsize=25, fontsize=20)
        if save:
            plt.savefig('comparison.pdf')
        plt.show()
