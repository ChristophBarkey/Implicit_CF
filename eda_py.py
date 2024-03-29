# %%
# %%
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class EDA:

    def __init__(self):
        pass

    # func to get the number of items each user has interacted with
    def items_per_user(self, csr):
        df = pd.DataFrame({'user' : csr.tocoo().row, 'item' : csr.tocoo().col})
        df_agg = df.groupby(by=['user']).count()
        return df_agg

    def get_user_per_item_frame(self, user_item_co, user_item_po):
        """" Function to get user per item info and vice versa for two user item matrices
        Initially used for comparing CO and PO, but also for CO of OEM1 and OEM2

        Parameters
        ----------
        user_item_1 : dataframe
            Pandas dataframe with columns: user, item, purchases
        user_item_2 : dataframe
            Pandas dataframe with columns: user, item, purchases
        
        Returns
        -------
        user_per_item_info : dataframe
            Pandas dataframe showing user per item info for both entities
        """
        po_items_per_user = user_item_po.groupby(by=['user']).count()
        co_items_per_user = user_item_co.groupby(by=['user']).count()
        po_users_per_item = user_item_po.groupby(by=['item']).count()
        co_users_per_item = user_item_co.groupby(by=['item']).count()
        podl_upi = po_users_per_item.user.describe()
        codl_upi = co_users_per_item.user.describe()
        podl_ipu = po_items_per_user['item'].describe()
        codl_ipu = co_items_per_user['item'].describe()
        return pd.DataFrame([podl_upi, codl_upi, podl_ipu, codl_ipu], index=['podl_upi', 'codl_upi', 'podl_ipu', 'codl_ipu'])

    def get_basic_user_item_info(self, user_item_co, user_item_po):
        """" Function to get basic information about two user item matrices
        Initially used for comparing CO and PO, but also for CO of OEM1 and OEM2

        Parameters
        ----------
        user_item_1 : dataframe
            Pandas dataframe with columns: user, item, purchases
        user_item_2 : dataframe
            Pandas dataframe with columns: user, item, purchases
        
        Returns
        -------
        user_item_info : dataframe
            Pandas dataframe showing user item info for both entities
        """
        sparsity_po = 1-(user_item_po.shape[0] / (user_item_po.user.nunique() * user_item_po['item'].nunique()))
        sparsity_co = 1-(user_item_co.shape[0] / (user_item_co.user.nunique() * user_item_co['item'].nunique()))
        nnz_po = len(user_item_po)
        nnz_co = len(user_item_co)
        nouser_po = user_item_po.user.nunique()
        nouser_co = user_item_co.user.nunique()
        noitem_po = user_item_po.item.nunique()
        noitem_co = user_item_co.item.nunique()
        co = [nouser_co, noitem_co, nnz_co, sparsity_co]
        po = [nouser_po, noitem_po, nnz_po, sparsity_po]
        return pd.DataFrame([co, po], index=['co', 'po'], columns=['nouser', 'noitem', 'nnz', 'sparsity'])


    def get_purchase_histograms(self, user_item_co_a, user_item_co_t, style='seaborn-whitegrid', color='#ae132a', 
    alpha=1, bins=30, title_fsize=25, label_fsize=15, ticks_fsize=10, size=(20, 8), save=False, scale_x='linear', scale_y='linear'):
        """" Function to generate histograms of purchase quantities for two user item matrices

        Parameters
        ----------
        user_item_1 : dataframe
            Pandas dataframe with columns: user, item, purchases
        user_item_2 : dataframe
            Pandas dataframe with columns: user, item, purchases
        style : str, optional
            Style of plot, default 'seaborn-whitegrid'
        color : str, optional
            Colorcode of bars, default '#ae132a'
        alpha : int, optional
            Percentage of transparency, default 1
        bins : int, optional
            Number of bins, default 30
        title_fsize : int, optional
            Fontsize of title
        label_size : int, optional
            Fontsize of labels
        ticks_fsize : int, optional
            Fontsize of ticknumbers
        size : tuple, int, optional
            Size of plot, (width, height)
        save : bool, optional
            Flag, whether to save the plot or not
        scale_x : str, optional
            Identifier of x-xaxis scale. Default 'lin' or 'log'
        scale_y : str, optional
            Identifier of y-xaxis scale. Default 'lin' or 'log'

        Returns
        -------
        histogram : plot
            Plot of two histograms
        """
        from matplotlib import pyplot as plt
        plt.style.use(style)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'cm'

        fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2)
        #ax.hist(pd.Series(user_item_csr_po.data), color='#ae132a', alpha=1)

        if isinstance(user_item_co_a, csr_matrix):
            data_a = user_item_co_a.data
        else:
            data_a = user_item_co_a.purchases

        ax[0].hist(data_a, color=color, alpha=alpha, bins=bins)
        ax[0].set_title('Distribution of purchase quantities OEM$\#$1', fontsize=title_fsize, pad=25)
        ax[0].set_xlabel('Number of purchases', fontsize=label_fsize)
        ax[0].set_ylabel('Frequency', fontsize=label_fsize)
        ax[0].tick_params(axis='both', which='major', labelsize=ticks_fsize)
        ax[0].tick_params(axis='both', which='minor', labelsize=ticks_fsize)

        if scale_x == 'log':
            ax[0].set_xscale('log')
        if scale_y == 'log':
            ax[0].set_yscale('log')

        if isinstance(user_item_co_t, csr_matrix):
            data_t = user_item_co_t.data
        else:
            data_t = user_item_co_t.purchases

        ax[1].hist(data_t, color=color, alpha=alpha, bins=bins)
        ax[1].set_title('Distribution of purchase quantities OEM$\#$2', fontsize=title_fsize, pad=25)
        ax[1].set_xlabel('Number of purchases', fontsize=label_fsize)
        ax[1].set_ylabel('Frequency', fontsize=label_fsize)
        ax[1].tick_params(axis='both', which='major', labelsize=ticks_fsize)
        ax[1].tick_params(axis='both', which='minor', labelsize=ticks_fsize)

        if scale_x == 'log':
            ax[1].set_xscale('log')
        if scale_y == 'log':
            ax[1].set_yscale('log')

        if save:
            plt.savefig('histogram_co.pdf', bbox_inches='tight')

        plt.show()

    def get_purchase_scatterplot(self, OEM, data, size=(28, 8), dpi=80, s=0.01, cmap='rainbow', save=False):
        from matplotlib import pyplot as plt
        if not isinstance(data, csr_matrix):
            data = self.to_csr(data)
        
        mtrx_dict = data.T.todok()
        xy = np.array(list(mtrx_dict.keys()))
        vals = np.array(list(mtrx_dict.values()))

        plt.figure(figsize=size, dpi=dpi)
        plt.scatter(xy[:,0], xy[:,1], s=s, c=vals, cmap=cmap)
        plt.colorbar()
        if save:
            if OEM == 'AGCO':
                plt.savefig('scatterplot_agco.pdf', bbox_inches='tight')
            if OEM == 'TEREX':
                plt.savefig('scatterplot_terex.pdf', bbox_inches='tight')
        plt.show()



