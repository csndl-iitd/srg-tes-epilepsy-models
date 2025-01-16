# Description of notebooks

### In-progress notebooks  

#### 1. Analyse Time of Entry 
#### 2. Analyse Node Communities
#### 3. Parallel Order Extraction


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



### Utility Notebooks  
Use these notebooks to know how to extract necessary data

#### 1. Automate Order Extraction
- Usage:
Computes global and local order data (for transition window of size 5000 timesteps). Stores in data folder.  
1. Extract Global Order Data
2. Extract Local Order Data
3. Store in data folder
4. Gives count of number of transitions

#### 2. Cluster Track
- Usage:
Plot cluster formation and extract node communities.  
1. Run cluster track  
2. Plot cluster  
3. Extract node communities


#### 3. Node Community Extraction
- Usage:  
Extract and store node community data for a given set of local order dataframes


### Test Notebooks
Utility notebooks are tested and verified
#### Test
Data used for test:  
1. Global Order   - itr_test.bz2  
2. Local Order    - loc_test.bz2
3. Node community - nc_test.bz2

#### 1. Test Automatic Order Extraction
#### 2. Test Node Community Extraction
#### 3. Test Parameter Algo