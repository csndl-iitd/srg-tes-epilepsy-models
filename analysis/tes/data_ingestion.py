from pathlib import Path
import pandas as pd
import pickle

class DataRead:
    """This class contains functions to read data from files.
        1. data_read_bz2(filename): To read .bz2 file
        2. data_read_pkl(filename): To read .pkl file
        3. data_read_csv(filename): To read .csv file

    """    

    main_data_dir = Path("D:\srg_tES\srg-tes-epilepsy-models\data")

    def data_read_bz2(self,filename):

        itr_path = self.main_data_dir / filename
        df=pd.read_pickle(itr_path,compression='bz2')
        return df

    def data_read_pkl(self,filename):
        pkl_path = self.main_data_dir / filename
        with open(pkl_path,'rb') as f:
            pkl = pickle.load(f)
        return pkl

    def data_read_csv(self, filename):
        csv_path = self.main_data_dir / filename
        df=pd.read_csv(csv_path,header=None)
        return df
