# Description of notebooks

### Analysis notebooks
Notebooks with name starting with 'Analyse' are for analysis purpose only. Listing analysis notebooks here:

#### 1. Analyse Global Order  
- Data used: 'itr_test.bz2'  

- Usage:  
All computation done for iteration number 5 in above data (one can change iteration number by changing value of n). This notebook does following:  
1. Smooth the data  
2. Calculate Transition Time  
3. Calculate Sync Time  
4. Calculate number of transitions
5. Plot global order vs time with crossing marked

#### 2. Analyse Parameter
- Data used: Server data
- Usage: 
Computes all parameters for all networks and do comparision plot
1. Compute transition time, sync time and number of transitions/100 iterations  
2. Plot transition time, sync time and number of transitions/100 iterations  
3. Store above parameters as pickle file





# Test
Data used for test:  
1. Global Order   - itr_test.bz2  
2. Local Order    - loc_test.bz2
3. Node community - nc_test.bz2