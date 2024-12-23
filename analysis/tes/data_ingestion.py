from pathlib import Path
import pandas as pd
import numpy as np
from tes.analysis import AnalysisFunc
from tes.ClusterTracking import SCTA
from tes.MBN_Res_Constrn import MBN_RC
import pickle


class DataRead:
    """This class contains functions to read data from files.
    1. data_read_bz2(filename): To read .bz2 file
    2. data_read_pkl(filename): To read .pkl file
    3. data_read_csv(filename): To read .csv file
    4. ext_node_community(df,th,filename): To extract node community for given local order data
    5. ext_global_order(itr_c,itr_filename,loc_filename): To extract global and local order automatically

    """
    # change main_data_dir path accordingly
    main_data_dir = Path("D:\srg_tES\srg-tes-epilepsy-models\data")

    def data_read_bz2(self, filename):

        itr_path = self.main_data_dir / filename
        df = pd.read_pickle(itr_path, compression="bz2")
        return df

    def data_read_pkl(self, filename):
        pkl_path = self.main_data_dir / filename
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        return pkl

    def data_read_csv(self, filename):
        csv_path = self.main_data_dir / filename
        df = pd.read_csv(csv_path, header=None)
        return df

    def ext_node_community(self, df, th, filename):
        """Extracts node community data and saves it as multi index dataframe in data folder.
        'level 0' index is iteration number. Plot global order data by using same column number as 'level 0'.
        MultiIndex dataframe is 5000x426 for 1 transition node community data.
        More on indexing in notebook 'Node Community Extraction.ipynb'

        Args:
            df (dataframe): MultiIndex dataframe of local order data with 'level 0' index as iteration number.
            th (float): Local Synchrony Order threshold
            filename (string): Name of file to store node community data
        """
        mbn = MBN_RC()
        # reset index values as it is a multi-index dataframe
        df_reset = df.reset_index()
        # reset outer index values i.e iteration numbers
        itr = df_reset["level_0"].drop_duplicates()
        # itr_val contains iteration numbers which had transitions
        itr_val = itr.values

        # df_nc to store node communities data
        df_nc = pd.DataFrame()
        tr_count = 0
        count = 0
        # put range of indices for how many transitions you need to do scta
        for x in itr_val:
            df_dum = df.loc[x].T
            if df_dum.shape[1] / 426 == 1:
                tr_count += 1
                local_order = df_dum.values
                scta = SCTA(
                    mbn.binary_conn, local_order, local_order_phase=None, sync_thrsh=th
                )
                scta.track_clusters()
                scta.update_synchronization_cluster_stats()
                df_ncdum = pd.DataFrame(scta.node_communities)

                # creating multi-index dataframe
                index = [[x] * 5000, list(range(0, 5000))]
                df_ncdum = df_ncdum.set_index(index)
                df_nc = pd.concat([df_nc, df_ncdum])
                print(f"{tr_count} Cluster track complete for transition {x}")
            else:

                f_limit = 426
                df_dum = df_dum.T
                # i is count of number of transitions
                i = df_dum.shape[0] / 426
                while count < i:
                    tr_count += 1
                    df_ext = df_dum.iloc[
                        0 + (f_limit * count) : 426 + (f_limit * count)
                    ]
                    df_ext = df_ext.T
                    local_order = df_ext.values
                    scta = SCTA(
                        mbn.binary_conn,
                        local_order,
                        local_order_phase=None,
                        sync_thrsh=th,
                    )
                    scta.track_clusters()
                    scta.update_synchronization_cluster_stats()
                    df_ncdum = pd.DataFrame(scta.node_communities)

                    # creating multi-index dataframe
                    index = [[x] * 5000, list(range(0, 5000))]
                    df_ncdum = df_ncdum.set_index(index)
                    df_nc = pd.concat([df_nc, df_ncdum])

                    print(f"{tr_count} Cluster track complete for transition {x}")

                    count += 1

        nc_path = self.main_data_dir / filename
        df_nc.to_pickle(nc_path, compression="bz2")
        print("Stored node communities data in data folder")

    def ext_global_order(self, itr_c, itr_filename, loc_filename):
        """Stores Global Order Data and Local Order Data for given number of simulations in data folder.
        Returns list of iteration numbers which had transitions.

        Args:
            itr_c (integer): Input count of simulations 
            itr_filename (string): filename (.bz2) to store global order data
            loc_filename (string): filename (.bz2) to store local order data

        Returns:
            list : Contains iteration numbers which had transition
        """    
        
        # create object of class AnalysisFunc
        analysis = AnalysisFunc()
        # create variable to count number of transitions
        trans_count=0
        # create list to store iteration number which have transitions
        itr_trans_num=[]

        # put itr as number of iterations you want
        itr = itr_c
        # to get and store iterations data
        c = 0

        # to store iteration data
        df_itr = pd.DataFrame()

        # to store local order data
        df_loc_data = pd.DataFrame()

        while c < itr:
            print("Current iteration is", c)
            mbn = MBN_RC(
                nepochs=40000,
                dt=0.05,
                lambda_o=2.86,
                alpha=0.01,
                beta=0.002,
                plot_bifurcation=False,
            )

            mbn.run_model()

            # df to store 1 iteration data
            df = pd.DataFrame(mbn.GLOBAL_ORDER_VERBOSE)
            # concatenating each iteration as a column
            df_itr = pd.concat([df_itr, df], axis=1)

            # changing headers to number of iterations
            df_itr.columns = range(0, df_itr.shape[1])
            # storing dataframe in csv file
            itr_path = self.main_data_dir / itr_filename
            df_itr.to_pickle(itr_path, compression="bz2")

            df_na = analysis.smooth(df)
            trans=analysis.trans_state_time(df_na,mbn.dt)
            if trans['num']>=1:
                data = np.array(df_na)

                # Extracting local order data for above iteration
                df_dum = pd.DataFrame()
                # defining local order data timestep thresholds

                # checking where it crossed 0.3
                m_loc = np.where(data >= 0.3)[0][0]

                lt_loc = m_loc - 3000
                ut_loc = m_loc + 2000

                # extracting data from df_loc
                header_list = list(range(lt_loc, ut_loc))
                # filtering those columns which lie in between 0 to mbn.nepochs
                header_list_f = [x for x in header_list if 0 <= x < mbn.nepochs]
                df_dum = mbn.df_loc[header_list_f]
                # changing column numbers so that they be concated 1 below other
                df_dum.columns = range(0, df_dum.shape[1])

                # creating multi-index dataframe
                index = [[c] * 426, list(range(0, 426))]
                df_dum = df_dum.set_index(index)
                df_loc_data = pd.concat([df_loc_data, df_dum])
                trans_count+=trans["num"]
                itr_trans_num.append(c)

                # increasing count by 1
                c += 1
            else:
                c+=1

        loc_path = self.main_data_dir / loc_filename
        df_loc_data = df_loc_data.astype(np.float32)
        df_loc_data.to_pickle(loc_path, compression="bz2")
        print('Stored Global Order data and Local order data in data folder')
        print(f'Number of transitions is {trans_count}')
        return itr_trans_num
