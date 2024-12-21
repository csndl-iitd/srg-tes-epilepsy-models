import pandas as pd
import numpy as np


class AnalysisFunc:
    """This class contains functions to use for analysis.
    1. smooth(df) to smooth global order data
    2. trans_state_time(df,dt) to calculate number of transitions, it's time and sync time

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
        """Counts the number of transitions, transition time and time spent in synchronised state. The lower threshold is 0.1 and upper threshold is 0.4.

        Args:
            df (dataframe): Smooth dataframe of global order
            dt (float): Timestep taken for the simulation

        Returns:
            dictionary: number of transitions, Transition time, Sync time if any
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
        keys = ["num", "trans_time", "state_time"]
        values = [num, trans_time, state_time]

        trans_dict = dict(zip(keys, values))
        return trans_dict
