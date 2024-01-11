""" 
Savanna Wolvin
Created: Feb 14th, 2023
Edited: Jun 2nd, 2023
    

##### SUMMARY #####
Script file holding the functions to formulate and save output statistics and 
models

##### FUNCTION LIST ##########################################################
    formulate_actual_predicted() - Function used to formulate the actual and 
                                    predicted values of OPG from the trained 
                                    CNN model
    stats() - Function to formulate the error, absolute error, relative error, 
                pearson correlation, r^2, spearman rank coefficient, and slope 
                between the actual OPG and predicted OPG 



"""
#%% Global Imports

import numpy as np
import postprocess_cnn_output as pp_cnn
import scipy.stats as spstats
import xarray as xr
import os




#%%
""" FUNCTION DEFINITION: formulate_actual_predicted
    INPUTS
    model       - Fitted Convolutional Neural Network
    atmos       - Xarray containing Atmospheric data
    opg         - Xarray containing OPG data
    opg_type    - What Type of OPG Values Was Used to Formulate Raw Values
    
    OUTPUT - Formulate actual VS predicted OPG values from the trained CNN model
"""

def formulate_actual_predicted(model, atmos, opg, opg_type):
    # Calculate Size of Data
    atm_sz = dict(atmos.sizes)
    opg_sz = dict(opg.sizes)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos.values():
        if len(ii.dims) == 3: count_channels += 1
        elif len(ii.dims) == 2: count_on_facet += 1
    
    # Create Empty Array for Atmos and OPG Data
    atmos_train_4D_X = np.zeros((atm_sz['time'], atm_sz['lat'], 
                                 atm_sz['lon'], count_channels))
    atmos_train_OF_X = np.zeros((atm_sz['time'], opg_sz['facet_num']*count_on_facet))
    
    # Pull Variables 
    count_4D = 0
    count_OF = 0
    for var_name, values in atmos.items():
        if len(values.dims) == 3:
            atmos_train_4D_X[:,:,:,count_4D] = np.array(values)
            count_4D += 1
        elif len(values.dims) == 2: 
            atmos_train_OF_X[:,opg_sz['facet_num']*count_OF:opg_sz['facet_num']*(count_OF+1)] = np.array(values)
            count_OF += 1
            
    # Combine Inputs if On-Facet Data Exists
    if count_on_facet > 0:
        inputs = [atmos_train_4D_X, atmos_train_OF_X]
    else:
        inputs = [atmos_train_4D_X]
    
    # Plot Heatmap of True VS Predicted
    predicted = np.squeeze(model.predict(inputs))
    actual = np.array(opg.opg.values)    
    
    # opg type presets
    if opg_type == 1:   
        actual_mm    = pp_cnn.standardized_to_raw(actual, opg)
        predicted_mm = pp_cnn.standardized_to_raw(predicted, opg)
    elif opg_type == 2: 
        actual_mm    = pp_cnn.normalized_to_raw(actual, opg)
        predicted_mm = pp_cnn.normalized_to_raw(predicted, opg)
        
    return actual_mm, predicted_mm, actual, predicted
    



#%%
""" FUNCTION DEFINITION: stats
    INPUTS
    save_dir    - Directory to Save the Stats
    model       - Fitted Convolutional Neural Network
    opg_type    - What Type of OPG Values Was Used to Formulate Raw Values
    name        - Name of the Predicted Data
    atmos       - Xarray containing Atmospheric data
    opg         - Xarray containing OPG data
    
    OUTPUT - Function to save model output statisitcs from the trained CNN
"""

def stats(save_dir, model, opg_type, name, atmos, opg):
    
    actual, predicted, actual_zscore, predicted_zscore = formulate_actual_predicted(
        model, atmos, opg, opg_type)
    
    f_num = len(opg.facet_num)
    t_num = len(atmos.time)

    r       = np.zeros((1, f_num)) # extent two variables are related
    r2      = np.zeros((1, f_num)) # percent of the variance that can be described in linear regression
    r_rank  = np.zeros((1, f_num)) # the strength and direction of the two variables
    slope   = np.zeros((1, f_num)) # the linear relationship between the actual and predicted
    e       = np.zeros((t_num, f_num)) # the difference between the actual and predicted
    ae      = np.zeros((t_num, f_num)) # the absolute difference between the actual and predicted
    re      = np.zeros((t_num, f_num)) # how large of the error relative to the actual value
    
    for facetx in range(len(opg.facet_num)):
        idx = actual[:,facetx] != 0.0
        
        cor = np.corrcoef(actual[idx,facetx], predicted[idx,facetx])
        r[:,facetx] = cor[0,1] # correlation
        r2[:,facetx] = cor[0,1]**2 # r-squared
        
        # r-spearman rank
        r_rank[:,facetx] = spstats.spearmanr(actual[idx,facetx], 
                                             predicted[idx,facetx]).correlation
        #slope
        m, b = np.polyfit(actual[idx,facetx], predicted[idx,facetx], 1)
        slope[:,facetx] = m
        
        # error
        e[idx,facetx] = (predicted[idx,facetx] - actual[idx,facetx])
        #e[idx,facetx] = (actual[idx,facetx] - predicted[idx,facetx])
        
        # absolute error
        ae[idx,facetx] = np.abs(predicted[idx,facetx] - actual[idx,facetx])
        #ae[idx,facetx] = np.abs(actual[idx,facetx] - predicted[idx,facetx])
        
        # relative error
        re[idx,facetx] = np.abs(predicted[idx,facetx] - actual[idx,facetx]) / actual[idx,facetx]
        #re[idx,facetx] = ((actual[idx,facetx] - predicted[idx,facetx])/predicted[idx,facetx])
    
        
    stats     = xr.Dataset()
    facet_num = opg.facet_num.values
    time      = atmos.time.values
    
    stats["error"]          = xr.DataArray(data = e,  coords = [time, facet_num], dims = ['time', 'facet_num'])
    stats["absolute_error"] = xr.DataArray(data = ae, coords = [time, facet_num], dims = ['time', 'facet_num'])
    stats["relative_error"] = xr.DataArray(data = re, coords = [time, facet_num], dims = ['time', 'facet_num'])
    
    stats["lin_stats"]      = xr.DataArray(data = np.concatenate([r,r2,r_rank,slope]), 
                                           coords = [["correlation", "r_squared", "r_rank", "slope"], facet_num], 
                                           dims = ['lin_stat', 'facet_num'])  
    
    stats["actual"]         = xr.DataArray(data = actual,    coords = [time, facet_num], dims = ['time', 'facet_num'])
    stats["predicted"]      = xr.DataArray(data = predicted, coords = [time, facet_num], dims = ['time', 'facet_num'])
    
    stats["actual_zscore"]         = xr.DataArray(data = actual_zscore,    coords = [time, facet_num], dims = ['time', 'facet_num'])
    stats["predicted_zscore"]      = xr.DataArray(data = predicted_zscore, coords = [time, facet_num], dims = ['time', 'facet_num'])
    
    path_stats = save_dir + "stats/"
    if os.path.isdir(path_stats) == False: os.mkdir(path_stats)
    stats.to_netcdf(path = path_stats + str(name) + "_output_stats.nc")
    
    
    
    