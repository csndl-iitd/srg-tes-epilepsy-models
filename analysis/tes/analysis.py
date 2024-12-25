import pandas as pd
import numpy as np


class AnalysisFunc:
    """This class contains functions to use for analysis.
    1. smooth(df) to smooth global order data
    2. trans_state_time(df,dt) to calculate number of transitions, it's time and sync time for agiven simulation
    3. compute_params(df,dt) to calculate above parameters for n iterations
    4. gives time units for X axis ticks to plot global order

    """

    def smooth(self, df):
        """This smooths global order data and drops NaN values so that easily thresholds could be spotted

        Args:
            df (dataframe): global synchrony order

        Returns:
            dataframe: smooth dataframe to work on
        """
        # Smooth the function
        df_smooth = df.rolling(window=800, center=True).mean()
        # Drop NaN values
        df_na = df_smooth.dropna()
        return df_na

    def trans_state_time(self, df, dt):
        """Counts the number of transitions, transition time and time spent in synchronised state for a given one global order dataframe.
        The lower threshold is 0.1 and upper threshold is 0.4.

        Args:
            df (dataframe): Smooth dataframe of global order
            dt (float): Timestep taken for the simulation

        Returns:
            dictionary: number of transitions 'num', Transition time 'trans_time', Avg Sync time 'state_time' if any
        """

        data = np.array(df)
        trans_time = []
        state_time = []

        while True:

            index_arr = np.where(data >= 0.4)[0]

            # Finding if iteration has a transition
            if index_arr.size > 0:
                upper_crossing = index_arr[0]
                # finding transition time thresholds
                l1 = np.where(data >= 0.1)[0]
                l2 = np.where(data <= 0.101)[0]
                low_intersection = np.intersect1d(l1, l2)
                # making sure lower thresholds are for 1st transition
                low_upd = low_intersection[low_intersection < upper_crossing]

                lower_crossing = low_upd[-1]

                t = (upper_crossing - lower_crossing) * dt
                trans_time.append(t)

                # checking for another transition
                fwd_data = np.array(data[upper_crossing:])

                check = np.where(fwd_data <= 0.2)[0]

                if check.size > 0:
                    ind = np.where(fwd_data <= 0.1)[0]
                    if ind.size > 0:

                        lower_crs = ind[0]
                        lower_crs_up = lower_crs + upper_crossing
                        st_time = (lower_crs_up - lower_crossing) * dt
                        state_time.append(st_time)
                        # creating data for other cycle of transition and state time
                        data = np.array(data[lower_crs_up:])

                    else:

                        break

                else:

                    break

            else:

                break
        # counts number of transitions
        num = len(trans_time)
        # Average state time
        if len(state_time) >= 1:
            state_time = sum(state_time) / len(state_time)

        keys = ["num", "trans_time", "state_time"]
        values = [num, trans_time, state_time]

        trans_dict = dict(zip(keys, values))
        return trans_dict

    def compute_param(self, df, dt):
        """
        Takes dataframe (timesteps x n) of global order for n itertaions as input and computes
        number of transition per 100 iteration, transition time and time spent in sync state.
        Also prints iteration number with transition times and sync time for exploration.

        Args:
            df (dataframe): timesteps x iteration count
            dt (timestep): simulation timestep

        Returns:
            dictionary: keys - ['tran_per','trans','sync']
        """
        i = 0
        num = 0
        tran_per = []
        trans = []
        sync = []
        itr_st = []
        itr_tran = []
        while i < df.shape[1]:
            df.columns = list(range(0, df.shape[1]))
            num_tran_sync = self.trans_state_time(self.smooth(df[i]), dt)
            num += num_tran_sync["num"]
            if i % 100 == 0 or i == df.shape[1] - 1:
                tran_per.append(num)
            if num_tran_sync["trans_time"]:
                itr_tran.append(i)
                for val in num_tran_sync["trans_time"]:
                    trans.append(val)
            if num_tran_sync["state_time"]:
                itr_st.append(i)
                sync.append(num_tran_sync["state_time"])
            i += 1
        sync_itr = dict(zip(itr_st, sync))
        tran_itr = dict(zip(itr_tran, trans))
        # print(trans)
        # print(sync)
        # print(num)
        print(tran_per)
        print(tran_itr)
        print(sync_itr)
        keys = ["tran_per", "trans", "sync"]
        values = [tran_per, trans, sync]
        param_dict = dict(zip(keys, values))
        return param_dict
    
    def get_time(self,df,dt):
        """Takes the global order dataframe, smooths it and gives time for X axis accordingly.
        Use it for plots.

        Args:
            df (dataframe): Global Order Dataframe timesteps x 1 eg. 40000 x 1
            dt (float): timestep

        Returns:
            list: time unit values for X axis
        """        
        df_smooth=self.smooth(df)
        time_steps_na = list(range(0,df_smooth.shape[0])) 
        time = np.multiply(time_steps_na,dt)
        return [time]

