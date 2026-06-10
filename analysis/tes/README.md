List of Python scripts with their usage:

1. analysis.py

   It contains preliminary functions which were initially used to explore data
  

2.  data_ingestion.py

   It contains class DataRead which contains following functions (primarily required to read, write and extract data from the model).

    1. data_read_bz2(filename): To read .bz2 file
    2. data_read_pkl(filename): To read .pkl file
    3. data_dump_pkl(filename,data): To dump .pkl file
    3. data_read_csv(filename): To read .csv file
    4. ext_node_community(df,th,filename): To extract node community for given local order data
    5. ext_global_order(itr_c,itr_filename,loc_filename): To extract global and local order automatically
    6. get_communities(): To get communnity map as pandas series
    7. new_algo_ext(): To get global order, local order using new algo

3. algo_partial_full.py
   It contains class to calculate parameters separately for partial and full transitions

4. algo_umap.py

   It contains class to identify transitions and do further analysis using pre-defined UMAP model

5. commAnalysis.py

   It contains class to compute community wise participation in different subtypes and further analysis
   
6.  DataUils.py , MBN_Res_Constrn.py and tES_Adaptive.py are model files.
7.  To add/remove nodes and find degree, use MBN_Res_Constrn.py file
8.  ClusterTracking.py - SCTA is implemented. Can be used to extract node community data as well as latest time of entry of nodes inside main cluster