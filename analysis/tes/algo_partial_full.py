import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import statistics
import pickle
from tes.algo_umap import UmapAlgo

from scipy.stats import gaussian_kde
algoumap = UmapAlgo()

class AlgoParFull:
    """This class contains functions to use for analysis and calculate various parameters for partial and full transitions separately.
    This will use UmapAlgo class from algo_umap.py.
    1. compute_coverage(df): Computes coverage of all visible transitions in an iteration and returns list containing coverage of each transition.
    2. compute_threshold(df): Compute threshold by plotting histogram and finding minima after smoothing it.
    """
    
    def compute_coverage(self, df, start, stop):
        """ [OUTDATED]Computes coverage for each transition in a given iteration.
            This uses global order data. New function 'compute_coverage_nc()' is already 
            defined in commAnalysis
        Args:
            df (dataframe): dataframe column of 1 x timestep entries
            start (list): From compute_parameters(df) in UmapAlgo
            stop (list): From compute_parameters(df) in UmapAlgo
        Returns:
            List: coverage of each transition"""

        cov_int = []
        df = df.rolling(800, center=True).mean().ffill().bfill()
        if not start:
            cov = 0
            cov_int.append(cov)

        elif not stop:
            cov = df[start[0] :].mean()
            cov_int.append(cov)

        elif len(start) == len(stop):
            i = 0
            while i < len(start):
                cov_int.append(df[start[i] : stop[i]].mean())
                i += 1

        else:
            i = 0
            stop.append(39999)  # as we simulate for 40000 timesteps
            while i < len(start):
                cov_int.append(df[start[i] : stop[i]].mean())
                i += 1

        return cov_int
    
    

    def compute_threshold(self, df):
        """ [OUTDATED] Computes threshold coverage to classify whether a transition is partial or full

        Args:
            df (dataframe): Whole dataset timesteps x total number of iterations
        Returns:
            float : Threshold coverage value"""
        algo = UmapAlgo()
        # iterating over whole dataset
        i = 0
        coverage = []
        while i < df.shape[1]:
            # print(i)
            # get start stop points
            start_stop = algo.compute_parameters(df[i])
            covr = self.compute_coverage(
                df[i], start_stop["start_points"], start_stop["stop_points"]
            )
            for j in covr:
                coverage.append(j)
            i += 1
        # removing all zero or NAN entries if any
        data = [k for k in coverage if k != 0 and not np.isnan(k)]
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 1000)
        y = kde(x)

        # Find local minima
        dy = np.gradient(y)
        minima_indices = np.where(np.diff(np.sign(dy)) > 0)[0]
        minima_x = x[minima_indices]

        # plt.plot(x, y)
        # plt.scatter(minima_x, y[minima_indices], color='red')
        # plt.title('KDE smooth of avg coverage histogram')
        # plt.show()

        # print("Estimated minima at:", minima_x)
        return minima_x[0]
    
    

    
    
    def compute_fpt(self, start, stop):
        """Computes average FPT for a particular iteration considering all transitions
        Args:
            start (list): From compute_parameters(df)
            stop (list): From compute_parameters(df)
        Return:
            list: All time units from unsync to sync state
        """
        fpt_int = []

        if not start:
            fpt = 0
            fpt_int.append(fpt)
        elif not stop:
            fpt = start[0]
            fpt_int.append(fpt)
        elif len(start) == len(stop):
            i = 1
            fpt_int.append(start[0])
            while i < len(start):
                fpt_int.append(start[i] - stop[i - 1])
                i += 1
            # print(fpt_int)

        else:
            i = 1
            fpt_int.append(start[0])
            while i < len(start):
                fpt_int.append(start[i] - stop[i - 1])
                i += 1
            # print(fpt_int)
            # fpt=statistics.mean(fpt_int)
        # multiply by dt=0.05
        fpt_int = [k * 0.05 for k in fpt_int]

        return fpt_int

    def compute_sync(self, start, stop):
        """Computes sync time for all the transitions in a given iteration (not considering the transition which never falls back).
            Make sure you remove all zero entries from the returned list.
        Args:
            start (list): From compute_parameters(df)
            stop (list): From compute_parameters(df)
        Return:
            list: All time units from sync to unsync state"""

        sync_int = []

        if not start:
            sync = 0
            sync_int.append(sync)

        elif not stop:
            sync = 0
            sync_int.append(sync)

        elif len(start) == len(stop):
            for i, j in zip(start, stop):
                sync_int.append(j - i + 800)

        else:
            start.pop()
            # print(start)
            for i, j in zip(start, stop):
                sync_int.append(j - i + 800)

        return [s_dum * 0.05 for s_dum in sync_int]

    def compute_num_trans(self, cov, threshold):
        """Classify transitions to partial and full and count them
        Args:
            cov (list): coverage of individual transitions in an iteration from compute_coverage()
            threshold (float): Threshold to classify if transition is partial or full
        Returns:
            dictionary: keys:('part_ind','full_ind','count_part','count_full','count_trans')
        """
        part_ind = []  # to store partial indices
        full_ind = []  # to store full transitions
        for ind, item in enumerate(cov):
            if item > threshold:
                full_ind.append(ind)

            else:
                part_ind.append(ind)

        return {
            "part_ind": part_ind,
            "full_ind": full_ind,
            "count_part": len(part_ind),
            "count_full": len(full_ind),
            "count_trans": len(part_ind) + len(full_ind),
        }

    def compute_param_df(self, df,network):
        """Computes every parameter for a dataset of particular NOI
        Args:
            df (dataframe): timesteps x number of iterations
            network (string): Make sure you already have threshold data stored
        Returns:
            Dictionary: 3 dictionaries"""
        algo = UmapAlgo()
        # thres = self.compute_threshold(df)
        from tes.data_ingestion import DataRead
        reader=DataRead()
        thres_data= reader.data_read_pkl('results/coverage_threshold.pkl')
        try:
            if network not in thres_data.keys():
                raise ValueError("Network not in saved threshold list")
            else:
                thres = thres_data[network]
                print(thres)
        except ValueError as e:
                print("Error:",e)


        #thres=0.39874500541287045  #fn
        # thres = 0.37842782078338566 #peri
        
        
        n = 0
        # to count partial transition
        tot_par_count = 0
        # to count full transition
        tot_full_count = 0
        # to store which iterations have transitions
        itr = []

        # to store parameters for partial transition
        p_trans_time = []
        p_fp_time = []
        p_sync_time = []
        p_cov = []

        # to store parameters for full transitions
        n_trans_time = []
        n_fp_time = []
        n_sync_time = []
        n_cov = []

        while n < df.shape[1]:
            print(n)
            k = algo.compute_parameters(df[n])
            # print(k)
            # check if transition or not
            if not k["start_points"]:
                # print("in not")
                n += 1
                continue
            else:
                itr.append(n)
                trans_time = algo.compute_transition_time(
                    df[n], k["start_points"].copy()
                )
                fp_time = self.compute_fpt(
                    k["start_points"].copy(), k["stop_points"].copy()
                )
                sync_time = self.compute_sync(
                    k["start_points"].copy(), k["stop_points"].copy()
                )
                cov = self.compute_coverage(
                    df[n], k["start_points"].copy(), k["stop_points"].copy()
                )
                trans_num = self.compute_num_trans(cov, thres)
                tot_par_count += trans_num["count_part"]
                tot_full_count += trans_num["count_full"]
                # print("completed params")

                if len(trans_num["part_ind"]) != 0:
                    for i in trans_num["part_ind"]:
                        if i < len(trans_time):
                            p_trans_time.append(trans_time[i])
                            if i < len(sync_time):
                                p_sync_time.append(sync_time[i])
                            p_fp_time.append(fp_time[i])
                            p_cov.append(cov[i])
                        else:
                            p_fp_time.append(fp_time[i])
                            p_cov.append(cov[i])
                    # print("part done")

                if len(trans_num["full_ind"]) != 0:
                    for i in trans_num["full_ind"]:
                        if i < len(trans_time):
                            n_trans_time.append(trans_time[i])
                            if i < len(sync_time):
                                n_sync_time.append(sync_time[i])
                            n_fp_time.append(fp_time[i])
                            n_cov.append(cov[i])
                        else:
                            n_fp_time.append(fp_time[i])
                            n_cov.append(cov[i])
                    # print("full done")
                n += 1

        # create dictionary to store no. of partial and full transition and iteration number which had transition
        dict_trans = {
            "n_part": tot_par_count,
            "n_full": tot_full_count,
            "n_total": tot_par_count + tot_full_count,
            "itr_tran": itr,
        }

        dict_par_params = {
            "t_t": p_trans_time,
            "fp_t": p_fp_time,
            "s_t": p_sync_time,
            "cov": p_cov,
        }

        dict_full_params = {
            "t_t": n_trans_time,
            "fp_t": n_fp_time,
            "s_t": n_sync_time,
            "cov": n_cov,
        }
        print("done")
        return dict_trans, dict_par_params, dict_full_params
