from pathlib import Path
import pandas as pd
from tes.ClusterTracking import SCTA
from tes.MBN_Res_Constrn import MBN_RC
import pickle


class DataRead:
    """This class contains functions to read data from files.
    1. data_read_bz2(filename): To read .bz2 file
    2. data_read_pkl(filename): To read .pkl file
    3. data_read_csv(filename): To read .csv file
    4. ext_node_community(df,th,filename): To extract node community for given local order data

    """

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
