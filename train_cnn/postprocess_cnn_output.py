""" 
Savanna Wolvin
Created: Nov 2nd, 2022
Edited: Mar 1st, 2023
    

##### SUMMARY #####
Script file holding the functions to post-process the output of predicted 
values from standardized OPG or normalized OPG.

##### FUNCTION LIST ##########################################################
    standardized_to_raw()
    normalized_to_raw()



"""
#%% Global Imports

import numpy as np




#%%
""" FUNCTION DEFINITION: standardized_to_raw
    INPUTS
    pred    - Array of Predicted OPG Values
    opg     - Xarray Containing Values for the Conversion
    
    OUTPUT - Convert Z-Score OPG Values to Original Values
"""

def standardized_to_raw(pred, opg):
    # save locations of zero values
    rows, cols = np.where(pred == 0)
    
    # Calc size of pred array
    sz = np.shape(pred)[0]
    
    # Pull Mean and Standard Deviation
    meanOPG = np.array(opg['mean_opg'])
    meanOPG = meanOPG[np.newaxis, :]
    stdOPG  = np.array(opg['std_opg'])
    stdOPG  = stdOPG[np.newaxis, :]
    
    # Reshape for multiplication
    meanOPG = np.repeat(meanOPG, sz, axis=0)
    stdOPG  = np.repeat(stdOPG,  sz, axis=0)
    
    # Calc OPG
    pred = (pred * stdOPG) + meanOPG
    
    # reset zero values
    pred[rows, cols] = 0
    
    return pred



#%%
""" FUNCTION DEFINITION: normalized_to_raw
    INPUTS
    pred    - Array of Predicted OPG Values
    opg     - Xarray Containing Values for the Conversion
    
    OUTPUT - Convert Normalized OPG Values to Original Values
"""

def normalized_to_raw(pred, opg):
    # save locations of zero values
    rows, cols = np.where(pred == 0)
    
    # Calc size of pred array
    sz = np.shape(pred)[0]
    
    # Pull min/max OPG
    minOPG = np.array(opg['min_opg'])
    minOPG = minOPG[np.newaxis, :]
    maxOPG = np.array(opg['max_opg'])
    maxOPG = maxOPG[np.newaxis, :]
    
    # Reshape for multiplication
    minOPG = np.repeat(minOPG, sz, axis=0)
    maxOPG = np.repeat(maxOPG, sz, axis=0)
    
    # Calc OPG
    pred = minOPG + (pred * (maxOPG - minOPG))
    
    # reset zero values
    pred[rows, cols] = 0
    
    return pred



