from pathlib import Path
import pandas as pd
import pickle


#change main_data_dir accordingly
main_data_dir = Path("D:\srg_tES\srg-tes-epilepsy-models\data")
def data_read_bz2(filename):

    itr_path = main_data_dir / filename
    df=pd.read_pickle(itr_path,compression='bz2')
    return df

def data_read_pkl(filename):
    pkl_path = main_data_dir / filename
    with open(pkl_path,'rb') as f:
        pkl = pickle.load(f)
    return pkl

def data_read_csv(filename):
    csv_path = main_data_dir / filename
    df=pd.read_csv(csv_path)
    return df
