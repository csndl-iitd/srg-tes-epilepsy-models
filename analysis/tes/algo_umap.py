import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import statistics
import pickle

class UmapAlgo:
    """ This class contains functions to use for analysis and calculate parameters
    using new algorithm - a) Number of transitions b) Transition time c) Sync state time d) Dwell time
    1. compute_parameters(df) : Takes 1 iteration data i.e 1xtimestep entries
    
    2. plot_transition(df,start,stop,chunk_size) : Plots global order and marks starting and ending of all perceivable transitions.
    
    3. compute_sync(start,stop) : Calculate sync state time by caculating average  time the network was in synchronised state
    
    4. compute_transition_time(df,start): Compute transition time given the start time for every transition
    
    5. compute_param_df(data): To compute all parameters for a set of transitions (timesteps x number of iteration)
    6. compute_coverage(df,start,stop): Computes average global order from 1st transition start to stop point
    """

    def compute_parameters(self,df): 
        """ Computes start and stop points of all the visible transitions in an iteration.
    
        Args: 
            df (dataframe): dataframe column of 1xtimesteps entries
        Returns: 
            Dictionary: Keys - 'start_points','stop_points'
        """
        # load pre-trained model 
        # change pathway when you finalise
        with open('../../data/umap_model.pkl', 'rb') as f:
            _s, _u = pickle.load(f)
        
        # smooth data
        _data=df.rolling(800, center=True).mean().ffill().bfill()
    
        chunk_size = 400
        # When you increase or decrease your chunk size, you alter the persistence i.e for how long it stays in sync state
        chunks = [_data[i:i + chunk_size] for i in range(0, len(_data)-chunk_size, chunk_size//4)]
        # create overlapping chunks with 75% overlap
        
        # _s = StandardScaler().fit(chunks)
        scaled_data = _s.transform(chunks)
        # Train StandardScaler, Z score normalising values
        
        # _u = u.fit(scaled_data)
        embedding = _u.transform(scaled_data)
        # Fit Umap i.e it converts 400 dimension data to 2 dimensions and we can create boundary
        
        sep = -0.6
    
        l_embed=[]
        for l in _data.index[:-chunk_size][::chunk_size//4][embedding[:, 0]>sep]:
            l_embed.append(l)
    
    # Finding start stop points of transition
        #print(l_embed)
        points_dict=self.compute_start_stop(df,l_embed)
        
        return points_dict

    
    def compute_start_stop(self,df,embed):
        """Compute start and stop points. 
        Args:
            embed (list): Contains timesteps
        Returns:
            Dictionary: Keys- 'start_points','stop_points'"""
        data=df.rolling(800, center=True).mean().ffill().bfill()
        # defining function to check if GO crosses 0.2
        def check_go(i):
            flag=False
            for val in embed[i:i+4]:
                
                if data[val]>0.2:
                    flag=True
                    
                    break
                else:
                    i+=1
            return flag

        def check_stop(j):
            
            #to compute stop
            flag_stop=False
            while j+1 < len(embed):
                if (embed[j+1]-embed[j])>400:
                    flag_stop=True
                    break
                else:
                    j+=1
            if flag_stop== False:
                if data[embed[j]]<0.3: 
                    flag_stop=True
                
            
            return [flag_stop,j] 
            
        
        i=0
        start=[]
        stop=[]
        while i+4 < len(embed):
            
            start_dum=embed[i]
            # check if consecutively 4 embeddings appear from start point
            if (embed[i+4] - start_dum)==400 and check_go(i)==True:
                start.append(embed[i])
                if check_stop(i)[0]==True:
                    stop.append(embed[check_stop(i)[1]])
                    i=check_stop(i)[1]+1
                else:
                    break
            else:
                i+=1
        return {
            'start_points':start,
            'stop_points':stop
        }                  
                        
                        
            
                
                
                
                
        
        
        
    def plot_transition(self,df,start,stop,chunk_size=400):
        """Plots the global order data and mark starting and ending of all transitions
        Args:
            df (dataframe): global synchrony order (1 x timestep)
            start, stop : Points from compute_parameters(df) function
            chunk_size  : By default it is 400
    
        Returns:
            None
            """
        _data=df.rolling(800, center=True).mean().ffill().bfill()
        fig,ax=plt.subplots(figsize=(12,3))
        cmap='jet'
        with open('umap_model.pkl', 'rb') as f:
            _s, _u = pickle.load(f) 
        chunk_size = 400
        # When you increase or decrease your chunk size, you alter the persistence i.e for how long it stays in sync state
        chunks = [_data[i:i + chunk_size] for i in range(0, len(_data)-chunk_size, chunk_size//4)]
        # create overlapping chunks with 75% overlap
        
        # _s = StandardScaler().fit(chunks)
        scaled_data = _s.transform(chunks)
        # Train StandardScaler, Z score normalising values
        
        # _u = u.fit(scaled_data)
        embedding = _u.transform(scaled_data)
        # Fit Umap i.e it converts 400 dimension data to 2 dimensions and we can create boundary
        
        sep = -0.6
        ax.scatter(_data.index, _data, c=_data.index, s=2, cmap=cmap)
        for i in stop:
            ax.plot(i+chunk_size,_data[i+chunk_size],'ko')
        for i in start: 
            ax.plot(i,_data[i],'mo')
        for l in _data.index[:-chunk_size][::chunk_size//4][embedding[:, 0]>sep]:
            ax.axvline(l, c='gray', alpha=0.1)
        ax.set_ylabel('Global Order')
        ax.set_xlabel('Time Steps')

    def compute_sync(self,start,stop): 
        """Calculate sync state time by calculating average  time the network was in synchronised state
        Args:
            start, stop : Take from compute_parameters(df)
        Returns:
            Float : Sync state time"""
        t=[]
        st=0
        chunk_size=400
        if len(start)!=0 and len(stop)!=0:
            if len(start)==len(stop) or len(start)>len(stop):
                for _start,_stop in zip(start,stop):
                    time = _stop+chunk_size-(_start-chunk_size)
                    t.append(time)
            # print(t)
            # multiply by 0.05 because it is our dt
            st=(sum(t)/len(t))*0.05
        return st

    def compute_transition_time(self,df,start): 
        """Compute transition time given the start time 
        Args:
            df (dataframe): single iteration 1 x timestep
            start (list): From compute_parameters(df)
        Returns:
            list: Containing transition time for every transition
        
        """
        window_size = 200
        _data=df.rolling(800, center=True).mean().ffill().bfill()
        # Define a function to calculate slope for a given window
        def calculate_slope(window):
            if len(window) < 2:  # Need at least two points to calculate slope
                return np.nan
            x = np.arange(0,len(window),1)
            y = window.values
            # Using numpy.polyfit for linear regression to get slope
            slope, intercept = np.polyfit(x, y, 1)
            return slope
        
        # Apply the slope calculation over a rolling window
        #rolling_slopes = _data[l_embed].rolling(window=window_size).apply(calculate_slope, raw=False)
        t_time=[]
        for k in start:
            rolling_slopes = _data[k:].rolling(window=window_size).apply(calculate_slope, raw=False)
            for i in rolling_slopes.index:
                if rolling_slopes[i] <0:
                    print(i)
                    c=i-1
                    t_time.append(c)
                    break
    
        
        t=[]
        if len(start)>0:
            for _init,_end in zip(start,t_time):
                time=_end-(_init-400)
                t.append(time)
        
        transition_time=[item*0.05 for item in t]
        
        return transition_time 

    def compute_param_df(self,data):
        """To compute all parameters for a set of transitions
        Args:
            data (dataframe): Global order data (timesteps x no. of iterations)
        Returns:
            dictionary: Keys: 'tran_per','sync','trans','fpt','cov'
            """
        c=0
        trans_count=0
        tran_per=[] # to store number of transitions per 100 iterations
        s_time=[]
        fpt=[]
        tran_time=[]
        coverage=[]
        while c<data.shape[1]: 
            l=self.compute_parameters(data[c])  # Calculate start and stop points
            
            trans_count=trans_count+len(l['start_points'])
            if c%100==0 or c==data.shape[1]-1:
                tran_per.append(trans_count)
    
            s_time.append(self.compute_sync(l['start_points'],l['stop_points']))
            if len(l['start_points'])!=0:
                fpt.append(self.compute_fpt(l['start_points'],l['stop_points']))
                
    
            t_time=self.compute_transition_time(data[c],l['start_points'])
            for item in t_time:
                tran_time.append(item)

            coverage.append(self.compute_coverage(data[c],l['start_points'],l['stop_points']))
            c+=1
        #remove zero coverage transitions
        cov=[j for j in coverage if j!=0 and not np.isnan(j)]
        sync=[i for i in s_time if i!=0]
        param_dict=dict(zip(['tran_per','sync','trans','fpt','cov'],[tran_per,sync,tran_time,fpt,cov]))
        return param_dict  

    def compute_coverage(self,df,start,stop):
        """ Computes average global order for a particular iteration considering all transitions
        Args:
            df (dataframe): Single iteration 1 x timestep
            start (list): From compute_parameters(df)
            stop (list): From compute_parameters(df)
        """
        df=df.rolling(800, center=True).mean().ffill().bfill()
        cov_int=[]
        if not start:
            cov=0
            #print('case1')
        elif not stop:
            cov= df[start[0]:].mean()
            #print('case2')
        elif len(start)==len(stop):
            i=0
            while i < len(start):
                cov_int.append(df[start[i]:stop[i]].mean())
                i+=1
            #print(cov_int)
            cov=statistics.mean(cov_int) 
            #print('case3')
        else:
            i=0
            # appending last timestep in stop
            stop.append(39999)
            while i < len(start):
                cov_int.append(df[start[i]:stop[i]].mean())
                i+=1
            cov=statistics.mean(cov_int)
            #print('case4')
        
        return cov

    def compute_fpt(self,start,stop):
        """Computes average FPT for a particular iteration considering all transitions
        Args:
            df (dataframe): Single iteration 1 x timestep
            start (list): From compute_parameters(df)
            stop (list): From compute_parameters(df)
        Return:
            fpt (float): Average FPT for an iteration
            """
        fpt_int=[]
        
        if not start:
            fpt=0
        elif not stop:
            fpt=start[0]
        elif len(start) == len(stop):
            i=1
            fpt_int.append(start[0])
            while i<len(start):
                fpt_int.append(start[i]-stop[i-1])
                i+=1
            #print(fpt_int)
            fpt=statistics.mean(fpt_int)
        else:
            i=1
            fpt_int.append(start[0])
            while i<len(start):
                fpt_int.append(start[i]-stop[i-1])
                i+=1
            #print(fpt_int)
            fpt=statistics.mean(fpt_int)
    
        return fpt*0.05