from tes.data_ingestion import DataRead
from tes.algo_umap import UmapAlgo
from tes.algo_partial_full import AlgoParFull
from tes.analysis import AnalysisFunc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pathlib import Path
import glob

reader = DataRead()
analyse=AnalysisFunc()
algo = AlgoParFull()
algoumap=UmapAlgo()

class CommPercentages:
    def __init__(self):
        self.full_mo_tot=[]
        self.full_mo_per=[]
        self.full_vis_tot=[]
        self.full_vis_per=[]
        self.full_orb_tot=[]
        self.full_orb_per=[] 
        self.full_hth_tot=[]
        self.full_hth_per=[]
        self.full_olf_tot=[]
        self.full_olf_per=[]
        self.full_hipp_tot=[]
        self.full_hipp_per=[] 
        self.full_mid_tot=[]
        self.full_mid_per=[]
        self.full_hind_tot=[]
        self.full_hind_per=[]

        self.part_mo_tot=[]
        self.part_mo_per=[]
        self.part_vis_tot=[]
        self.part_vis_per=[]
        self.part_orb_tot=[]
        self.part_orb_per=[] 
        self.part_hth_tot=[]
        self.part_hth_per=[]
        self.part_olf_tot=[]
        self.part_olf_per=[]
        self.part_hipp_tot=[]
        self.part_hipp_per=[] 
        self.part_mid_tot=[]
        self.part_mid_per=[]
        self.part_hind_tot=[]
        self.part_hind_per=[]

    
    def get_crossing_go(self,df, pts):
        """Get start and stop points where global order is 90%

        Args:
            df (dataframe): Global order dataframe of timesteps x iterations
            pts (dictionary): Start and Stop points of transition
            

        Returns:
            dictionary: keys are 'start_comm' and 'stop_comm' with respective lists of points
        """
        # Ensure stop points match start points
        if len(pts['start_points']) > len(pts['stop_points']):
            pts['stop_points'].append(len(df) - 1)

        _data = df.rolling(800, center=True).mean().ffill().bfill()
        start_comm=[]
        stop_comm=[]
        thresh_go=[]
        all_crossings = []
        for start, stop in zip(pts['start_points'], pts['stop_points']):
            #_data = df_fn[n].iloc[start:stop].rolling(800, center=True).mean().ffill().bfill()
            values = _data.iloc[start:stop].values
            window_values = values.copy()

            max_val = values.max()
            threshold = 0.90 * max_val
            thresh_go.append(threshold)
            above_threshold = window_values >= threshold

            # Detect transitions
            transitions = np.diff(above_threshold.astype(int))
            rising_edges = np.where(transitions == 1)[0]      # below → above
            falling_edges = np.where(transitions == -1)[0]     # above → below

            # Combine and sort all crossings
            #all_edges = np.sort(np.concatenate((rising_edges, falling_edges)))
            all_edges = np.concatenate((rising_edges, falling_edges))
            actual_indices = all_edges + start

            if len(actual_indices) > 0:
                all_crossings.extend(actual_indices)
                start_comm.append(actual_indices[0])
                # stop_comm.append(actual_indices[-1])
                if actual_indices[0]!=actual_indices[-1]:
                    stop_comm.append(actual_indices[-1])
                elif stop==39999:
                    # to take care of transitions which never fall back
                    stop_comm.append(stop)
                else:
                    # to take care of rolling window in the end cases
                    print('this cell')
                    stop_comm.append(stop+400)

                #print(f'Start {start} Stop {stop} Threshold {threshold}')
                #print(f'Crossing points (rising & falling): {actual_indices}')
            else:
                #print(f'No threshold crossing found between {start} and {stop}')
                closest_idx = np.argmin(np.abs(window_values - threshold))
                actual_idx = int(closest_idx) + start
                #print(f'Closest point to threshold at index: {actual_idx}')
                start_comm.append(start)
                stop_comm.append(stop)

        # print(f'\nTotal number of crossing points found: {len(all_crossings)}')
        # print(f' points {start_comm} ,{stop_comm}, {thresh_go}')
        
        return {'start_comm':start_comm,'stop_comm':stop_comm}


    def get_binairized_matrix(self,df,trans_start):
        """Gives binarized matrix in desired order. YOu can uncomment plt lines to visualise the matrix.

        Args:
            df (dataframe): node community dataframe only transition part (5000x426)
            trans_start (int): starting point of transition
        Returns:
            dataframe: Binarized dataframe with community ordering
        """
        des_order=reader.get_communities()['des_order']
        mc = analyse.compute_main_cluster(df)
        nc_array = df.values
        nc=nc_array.copy()
        #print(mc)

        binary_mask = np.isclose(nc, mc)
        bm_ordered = binary_mask.T[des_order]

        df_bm=pd.DataFrame(bm_ordered).T
        #trans_start=pts['start_points'][0]
        df_bm.index=list(range(trans_start-3000,trans_start+2000))
        # plt.imshow(df_bm.T, cmap='gray', interpolation='nearest', aspect = 'auto')
        # plt.colorbar()
        # plt.title('Binarised matrix with community ordering')
        return df_bm



    def get_community_percentage(self,start, stop, df_bm):
        """
        Calculates community-wise percentage for a given transition range.

        Args:
            start (int/float): starting timestep (e.g., 90% global order timestep)
            stop (int/float): ending timestep (e.g., below 90% global order timestep)
            df_bm (pd.DataFrame): binarised node-community dataframe (timesteps × nodes)
            
        """
        comm_count=reader.get_communities()['comm_val']
        comm_name=reader.get_communities()['name_seq']
        cum_count=np.cumsum(comm_count)

        # Define the time window (handle missing stop)
        if start not in df_bm.index:
            raise ValueError("Start index not found in dataframe")
        if stop in df_bm.index:
            df_slice = df_bm.loc[start:stop]
        else:
            df_slice = df_bm.loc[start:]
        
        results = []

        init = 0
        for count, name in zip(cum_count, comm_name):
            sub_array = df_slice.iloc[:, init:count].values
            tot=sub_array.size
            count_true=np.sum(sub_array)
            percentage=(count_true/tot)*100
            results.append({
                "community": name,
                "total": tot,
                "percentage": percentage
            })
            init = count
        df=pd.DataFrame(results)
        df=df.set_index('community')
        # return df_slice to get overall coverage
        return df, df_slice



    def store_full_data(self, df_gc):
        self.full_mo_tot.append(df_gc.loc['MO']['total'])
        self.full_mo_per.append(df_gc.loc['MO']['percentage'])

            
        self.full_vis_tot.append(df_gc.loc['VIS']['total'])
        self.full_vis_per.append(df_gc.loc['VIS']['percentage'])

                
        self.full_orb_tot.append(df_gc.loc['ORB']['total'])
        self.full_orb_per.append(df_gc.loc['ORB']['percentage'])


        self.full_hth_tot.append(df_gc.loc['HTh']['total'])
        self.full_hth_per.append(df_gc.loc['HTh']['percentage'])

        
        self.full_olf_tot.append(df_gc.loc['OLF']['total'])
        self.full_olf_per.append(df_gc.loc['OLF']['percentage'])
        print(f'full: {self.full_olf_per}')

        self.full_hipp_tot.append(df_gc.loc['HIPP']['total'])
        self.full_hipp_per.append(df_gc.loc['HIPP']['percentage'])


        self.full_mid_tot.append(df_gc.loc['MID']['total'])
        self.full_mid_per.append(df_gc.loc['MID']['percentage'])


        self.full_hind_tot.append(df_gc.loc['HIND']['total'])
        self.full_hind_per.append(df_gc.loc['HIND']['percentage'])

    def store_part_data(self, df_gc):
        self.part_mo_tot.append(df_gc.loc['MO']['total'])
        self.part_mo_per.append(df_gc.loc['MO']['percentage'])

            
        self.part_vis_tot.append(df_gc.loc['VIS']['total'])
        self.part_vis_per.append(df_gc.loc['VIS']['percentage'])

                
        self.part_orb_tot.append(df_gc.loc['ORB']['total'])
        self.part_orb_per.append(df_gc.loc['ORB']['percentage'])


        self.part_hth_tot.append(df_gc.loc['HTh']['total'])
        self.part_hth_per.append(df_gc.loc['HTh']['percentage'])

        
        self.part_olf_tot.append(df_gc.loc['OLF']['total'])
        self.part_olf_per.append(df_gc.loc['OLF']['percentage'])
        print(f'part: {self.part_olf_per}')


        self.part_hipp_tot.append(df_gc.loc['HIPP']['total'])
        self.part_hipp_per.append(df_gc.loc['HIPP']['percentage'])


        self.part_mid_tot.append(df_gc.loc['MID']['total'])
        self.part_mid_per.append(df_gc.loc['MID']['percentage'])


        self.part_hind_tot.append(df_gc.loc['HIND']['total'])
        self.part_hind_per.append(df_gc.loc['HIND']['percentage'])

    def weighted_avg_and_std(self, values, weights):
        # Convert to NumPy arrays to allow element-wise arithmetic
        values = np.array(values, dtype=float)
        weights = np.array(weights, dtype=float)

        # Weighted mean
        average = np.average(values, weights=weights)

        # Unbiased weighted variance (sample version)
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights * weights)  # element-wise multiplication
        variance = np.sum(weights * (values - average)**2) / (sum_w - (sum_w2 / sum_w))
        # std = np.sqrt(variance)

        # Effective sample size
        n_eff = sum_w**2 / sum_w2

        # Weighted SEM
        sem = np.sqrt(variance) / np.sqrt(n_eff)

        return average, sem



    def compute_community_participation(self):
        # print(f'Comunity wise participation for Full transiions')
        # print(f'The average MO participation : {np.average(self.full_mo_per,weights=self.full_mo_tot)}')
        # print(f'The average VIS participation : {np.average(self.full_vis_per,weights=self.full_vis_tot)}')
        # print(f'The average ORB participation : {np.average(self.full_orb_per,weights=self.full_orb_tot)}')
        # print(f'The average HTh participation : {np.average(self.full_hth_per,weights=self.full_hth_tot)}')
        # print(f'The average OLF participation : {np.average(self.full_olf_per,weights=self.full_olf_tot)}') 
        # print(f'The average HIND participation : {np.average(self.full_hind_per,weights=self.full_hind_tot)}')
        # print(f'The average HIPP participation : {np.average(self.full_hipp_per,weights=self.full_hipp_tot)}')
        # print(f'The average MID participation : {np.average(self.full_mid_per,weights=self.full_mid_tot)}')

        
        # print(f'Comunity wise participation for Partial transiions')
        # print(f'The average MO participation : {np.average(self.part_mo_per,weights=self.part_mo_tot)}')
        # print(f'The average VIS participation : {np.average(self.part_vis_per,weights=self.part_vis_tot)}')
        # print(f'The average ORB participation : {np.average(self.part_orb_per,weights=self.part_orb_tot)}')
        # print(f'The average HTh participation : {np.average(self.part_hth_per,weights=self.part_hth_tot)}')
        # print(f'The average OLF participation : {np.average(self.part_olf_per,weights=self.part_olf_tot)}')
        # print(f'The average HIND participation : {np.average(self.part_hind_per,weights=self.part_hind_tot)}')
        # print(f'The average HIPP participation : {np.average(self.part_hipp_per,weights=self.part_hipp_tot)}')
        # print(f'The average MID participation : {np.average(self.part_mid_per,weights=self.part_mid_tot)}') 
        
        # avg_full_mo = np.average(self.full_mo_per,weights=self.full_mo_tot)
        # avg_full_vis = np.average(self.full_vis_per,weights=self.full_vis_tot)
        # avg_full_orb = np.average(self.full_orb_per,weights=self.full_orb_tot)
        # avg_full_hth = np.average(self.full_hth_per,weights=self.full_hth_tot)
        # avg_full_olf = np.average(self.full_olf_per,weights=self.full_olf_tot)
        # avg_full_hind = np.average(self.full_hind_per,weights=self.full_hind_tot)
        # avg_full_hipp = np.average(self.full_hipp_per,weights=self.full_hipp_tot)
        # avg_full_mid = np.average(self.full_mid_per,weights=self.full_mid_tot)      

        c_mo_per = [x for x in self.part_mo_per if x==x]
        c_vis_per = [x for x in self.part_vis_per if x==x]
        c_orb_per = [x for x in self.part_orb_per if x==x]
        c_hth_per = [x for x in self.part_hth_per if x==x]
        c_olf_per = [x for x in self.part_olf_per if x==x]
        c_hind_per = [x for x in self.part_hind_per if x==x]
        c_hipp_per = [x for x in self.part_hipp_per if x==x]
        c_mid_per = [x for x in self.part_mid_per if x==x]

        c_mo_tot= [x for x in self.part_mo_tot if x!=0]
        c_vis_tot= [x for x in self.part_vis_tot if x!=0]
        c_orb_tot= [x for x in self.part_orb_tot if x!=0]
        c_hth_tot= [x for x in self.part_hth_tot if x!=0]
        c_olf_tot= [x for x in self.part_olf_tot if x!=0]
        c_hind_tot= [x for x in self.part_hind_tot if x!=0]
        c_hipp_tot= [x for x in self.part_hipp_tot if x!=0]
        c_mid_tot= [x for x in self.part_mid_tot if x!=0]       

        avg_part_mo = np.average(c_mo_per,weights=c_mo_tot)
        avg_part_vis = np.average(c_vis_per,weights=c_vis_tot)
        avg_part_orb = np.average(c_orb_per,weights=c_orb_tot)
        avg_part_hth = np.average(c_hth_per,weights=c_hth_tot)
        avg_part_olf = np.average(c_olf_per,weights=c_olf_tot)
        avg_part_hind = np.average(c_hind_per,weights=c_hind_tot)
        avg_part_hipp = np.average(c_hipp_per,weights=c_hipp_tot)
        avg_part_mid = np.average(c_mid_per,weights=c_mid_tot)

        # FULL transitions
        avg_full_mo, sem_full_mo = self.weighted_avg_and_std(self.full_mo_per, self.full_mo_tot)
        avg_full_vis, sem_full_vis = self.weighted_avg_and_std(self.full_vis_per, self.full_vis_tot)
        avg_full_orb, sem_full_orb = self.weighted_avg_and_std(self.full_orb_per, self.full_orb_tot)
        avg_full_hth, sem_full_hth = self.weighted_avg_and_std(self.full_hth_per, self.full_hth_tot)
        avg_full_olf, sem_full_olf = self.weighted_avg_and_std(self.full_olf_per, self.full_olf_tot)
        avg_full_hind, sem_full_hind = self.weighted_avg_and_std(self.full_hind_per, self.full_hind_tot)
        avg_full_hipp, sem_full_hipp = self.weighted_avg_and_std(self.full_hipp_per, self.full_hipp_tot)
        avg_full_mid, sem_full_mid = self.weighted_avg_and_std(self.full_mid_per, self.full_mid_tot)

        # PARTIAL transitions
        avg_part_mo, sem_part_mo = self.weighted_avg_and_std(c_mo_per, c_mo_tot)
        avg_part_vis, sem_part_vis = self.weighted_avg_and_std(c_vis_per, c_vis_tot)
        avg_part_orb, sem_part_orb = self.weighted_avg_and_std(c_orb_per, c_orb_tot)
        avg_part_hth, sem_part_hth = self.weighted_avg_and_std(c_hth_per, c_hth_tot)
        avg_part_olf, sem_part_olf = self.weighted_avg_and_std(c_olf_per, c_olf_tot)
        avg_part_hind, sem_part_hind = self.weighted_avg_and_std(c_hind_per, c_hind_tot)
        avg_part_hipp, sem_part_hipp = self.weighted_avg_and_std(c_hipp_per, c_hipp_tot)
        avg_part_mid, sem_part_mid = self.weighted_avg_and_std(c_mid_per, c_mid_tot)


        return {'avg_full':{'MO':avg_full_mo,'VIS':avg_full_vis,'ORB':avg_full_orb,'HTh':avg_full_hth,'OLF':avg_full_olf,'HIPP':avg_full_hipp,'MID':avg_full_mid,'HIND':avg_full_hind},
                'avg_part':{'MO':avg_part_mo,'VIS':avg_part_vis,'ORB':avg_part_orb,'HTh':avg_part_hth,'OLF':avg_part_olf,'HIPP':avg_part_hipp,'MID':avg_part_mid,'HIND':avg_part_hind},
                'sem_full':{'MO':sem_full_mo,'VIS':sem_full_vis,'ORB':sem_full_orb,'HTh':sem_full_hth,'OLF':sem_full_olf,'HIPP':sem_full_hipp,'MID':sem_full_mid,'HIND':sem_full_hind},
                'sem_part':{'MO':sem_part_mo,'VIS':sem_part_vis,'ORB':sem_part_orb,'HTh':sem_part_hth,'OLF':sem_part_olf,'HIPP':sem_part_hipp,'MID':sem_part_mid,'HIND':sem_part_hind}
                }
        
    def get_avg_comm_particptn(self,df,df_nc):
    

        tran_itr=algo.compute_param_df(df)[0]['itr_tran']
        #tran_itr=[0, 1, 2, 5,]
        threshold = algo.compute_threshold(df)
        #threshold=0.39874500541287045 
        
        for itr in tran_itr:
            # find start stop points of transition
            pts=algoumap.compute_parameters(df[itr])
            
            # find itr which have partial or full transitions
            cov=algo.compute_coverage(df[itr], pts['start_points'], pts['stop_points'])
            ind=algo.compute_num_trans(cov,threshold)
            part_ind=ind['part_ind']
            full_ind=ind['full_ind']
            
            # Find 90% go crossing
            comm_pts=self.get_crossing_go(df[itr], pts)
            
            # iterating over node community data
            nc_lim=df_nc.loc[itr].shape[0]/5000
            ind=list(range(0,int(nc_lim)))

            i=0
            c=5000
            while i<len(ind):
                df_dum=df_nc.loc[itr].iloc[i*5000:c]
                df_bm=self.get_binairized_matrix(df_dum,comm_pts['start_comm'][i])
                df_gc=self.get_community_percentage( comm_pts['start_comm'][i], comm_pts['stop_comm'][i], df_bm)
                if i in part_ind:
                    #print(df_gc)
                    
                    self.store_part_data(df_gc)
                    print(itr,i)

                else:
                    #print(df_gc)
                    self.store_full_data(df_gc)
                    print(itr,i)

                c+=5000
                i+=1

    def compute_coverage_nc(self,df,df_nc):
        tran_itr=algo.compute_param_df(df)[0]['itr_tran']
        total_coverage=[]

        for itr in tran_itr:
            pts = algoumap.compute_parameters(df[itr])
            comm_pts = self.get_crossing_go(df[itr],pts)
            # algoumap.plot_transition(df[itr], pts['start_points'],pts['stop_points'])
            # iterating over node community data
            nc_lim=df_nc.loc[itr].shape[0]/5000
            ind=list(range(0,int(nc_lim)))

            i=0
            c=5000
            while i<len(ind):
                df_dum=df_nc.loc[itr].iloc[i*5000:c]
                df_bm=self.get_binairized_matrix(df_dum,comm_pts['start_comm'][i])
                cov_nodes=self.get_community_percentage( comm_pts['start_comm'][i], comm_pts['stop_comm'][i], df_bm)
                # Counting nodes in main cluster
                node_mc =cov_nodes[1].values.sum()
                total_nodes = cov_nodes[1].size
                coverage = node_mc / total_nodes
                total_coverage.append(coverage)
                c+=5000
                i+=1

        # finding minima
        data=total_coverage
        data = [k for k in data if k != 0 and not np.isnan(k)]
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 1000)
        y = kde(x)

        # Find local minima
        dy = np.gradient(y)
        minima_indices = np.where(np.diff(np.sign(dy)) > 0)[0]
        minima_x = x[minima_indices]

        
        return [total_coverage,minima_x[0]]

    def compute_individual_coverage_nc(self,df_nc_itr,df_go_itr):
        """Compute coverage for a particular iteration.

        Args:
            df_nc_itr (dataframe): node community dataframe of iteration
            df_go_itr (dataframe): global order dataframe of iteration (1 x timesteps)

        Returns:
            float: Coverage of that particular iteration
        """

        pts = algoumap.compute_parameters(df_go_itr)
        comm_pts = self.get_crossing_go(df_go_itr,pts)
        # algoumap.plot_transition(df[itr], pts['start_points'],pts['stop_points'])
        # iterating over node community data
        nc_lim=df_nc_itr.shape[0]/5000
        ind=list(range(0,int(nc_lim)))
        total_coverage=[]
        i=0
        c=5000
        while i<len(ind):
            df_dum=df_nc_itr.iloc[i*5000:c]
            df_bm=self.get_binairized_matrix(df_dum,comm_pts['start_comm'][i])
            cov_nodes=self.get_community_percentage( comm_pts['start_comm'][i], comm_pts['stop_comm'][i], df_bm)
            # Counting nodes in main cluster
            node_mc =cov_nodes[1].values.sum()
            total_nodes = cov_nodes[1].size
            coverage = node_mc / total_nodes
            total_coverage.append(coverage)
            c+=5000
            i+=1

        
        return total_coverage
             

        
      









