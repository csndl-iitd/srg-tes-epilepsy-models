List of Python scripts with their usage:

1. analysis.py

   It contains class AnalysisFunc which contains following functions (primarily required for analysis).
    1. smooth(df) to smooth global order data
    2. trans_state_time(df,dt) to calculate number of transitions, it's time and sync time for agiven simulation
    3. compute_params(df,dt) to calculate above parameters for n iterations
    4. get_time(df,dt) to get time units for X axis ticks to plot global order
    5. compute_toe(nc,main_cluster) to get time of entry of nodes
    6. plot_global_synchrony(df,dt) to plot global order wrt time
  

2.  data_ingestion.py

   It contains class DataRead which contains following functions (primarily required to read, write and extract data from the model).

    1. data_read_bz2(filename): To read .bz2 file
    2. data_read_pkl(filename): To read .pkl file
    3. data_dump_pkl(filename,data): To dump .pkl file
    3. data_read_csv(filename): To read .csv file
    4. ext_node_community(df,th,filename): To extract node community for given local order data
    5. ext_global_order(itr_c,itr_filename,loc_filename): To extract global and local order automatically
    6. get_communities(): To get communnity map as pandas series

3.  DataUils.py , MBN_Res_Constrn.py and tES_Adaptive.py are model files.
4.  To add/remove nodes and find degree, use MBN_Res_Constrn.py file
5.  ClusterTracking.py - SCTA is implemented. Can be used to extract node community data as well as latest time of entry of nodes inside main cluster