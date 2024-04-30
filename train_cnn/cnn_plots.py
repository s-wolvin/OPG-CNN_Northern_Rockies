""" 
Savanna Wolvin
Created: Sep 15th, 2022
Edited: Apr 30th, 2024
    

##### SUMMARY #####
Script file holding the functions to create numerous types of plots for
cnn_single_facet_MAIN.py

##### FUNCTION LIST ##########################################################
    formulate_actual_predicted() - Function used to formulate the actual and 
                                    predicted values of OPG from the trained 
                                    CNN model
    hist_r2_rrank_MAE_slope() - Plot histograms of the r^2, spearman rank 
                                    coefficient, mean absolute error, and slope 
                                    between actual OPG and predicted OPG 
    heatmap_by_facet() - Plot Heatmap of True OPG VS Predicted OPG by Facet
    heatmap() - Plot Heatmap of True OPG VS Predicted OPG
    training_loss() - Plot the Training Loss from Training the CNN
    training_validation_loss() - Plot the Training and Validation Loss from 
                                    Training the CNN



"""
#%% Global Imports

import numpy as np
import matplotlib.pyplot as plt
import nclcmaps as ncm
from matplotlib import colors
import postprocess_cnn_output as pp_cnn
import scipy.stats as spstats
import os




#%%
""" FUNCTION DEFINITION: formulate_actual_predicted
    INPUTS
    model       - Fitted Convolutional Neural Network
    atmos       - Xarray containing Atmospheric data
    opg         - Xarray containing OPG data
    opg_type    - What Type of OPG Values Was Used to Formulate Raw Values
    
    OUTPUT - Heatmap Plot of True OPG Values VS Predicted OPG Values by Facet
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
        actual    = pp_cnn.standardized_to_raw(actual, opg)
        predicted = pp_cnn.standardized_to_raw(predicted, opg)
    elif opg_type == 2: 
        actual    = pp_cnn.normalized_to_raw(actual, opg)
        predicted = pp_cnn.normalized_to_raw(predicted, opg)
        
    return actual, predicted
    


#%% 
""" FUNCTION DEFINITION: hist_r2_rrank_MAE_slope
    INPUTS
    save_dir    - Directory to Save the Figure
    model       - Fitted Convolutional Neural Network
    opg_type    - What Type of OPG Values Was Used to Formulate Raw Values
    name        - Name of the Predicted Data
    atmos       - Xarray containing Atmospheric data
    opg         - Xarray containing OPG data
    
    OUTPUT - Heatmap Plot of True OPG Values VS Predicted OPG Values by Facet
"""

def hist_r2_rrank_MAE_slope(save_dir, model, opg_type, name, atmos, opg):
    
    actual, predicted = formulate_actual_predicted(model, atmos, opg, opg_type)

    units = "mm/km"

    r2 = np.zeros((len(opg.facet_num),1))
    r_rank = np.zeros((len(opg.facet_num),1))
    slope = np.zeros((len(opg.facet_num),1))
    mae = np.zeros((len(opg.facet_num),1))
    me = np.zeros((len(opg.facet_num),1))
    
    for facetx in range(len(opg.facet_num)):
        idx = actual[:,facetx] != 0.0
        # r-squared
        cor = np.corrcoef(actual[idx,facetx], predicted[idx,facetx])**2
        r2[facetx,:] = cor[0,1]
        # r-spearman rank
        r_rank[facetx,:] = spstats.spearmanr(actual[idx,facetx], 
                                             predicted[idx,facetx]).correlation
        #slope
        m, b = np.polyfit(actual[idx,facetx], predicted[idx,facetx], 1)
        slope[facetx,:] = m
        
        # mean absolute error        
        mae[facetx,:] = np.nanmean(np.abs(actual[idx,facetx] - predicted[idx,facetx]))
        
        # mean error        
        me[facetx,:] = np.nanmean((actual[idx,facetx] - predicted[idx,facetx]))
    
    # plot histogram
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle("Actual VS Predicted OPG of " + name + " Data", fontsize=18)
    
    # r-squared
    n1,_,_ = axs[0,0].hist(r2, bins=np.arange(0,1.05,0.05), 
                edgecolor="black", facecolor="lightcoral")
    axs[0,0].set_title("$\mathregular{r^2}$", fontsize=14, weight='bold')
    axs[0,0].set(xlim=(0, 1))
    axs[0,0].grid(True)
    axs[0,0].set_ylabel("Count of Facets", fontsize=12)
    axs[0,0].tick_params(labelsize=12)
    
    # slope
    n2,_,_ = axs[0,1].hist(slope, bins=np.arange(0,1.05,0.05), 
                edgecolor="black", facecolor="orangered")
    axs[0,1].set_title("Slope", fontsize=14, weight='bold')
    axs[0,1].set(xlim=(0, 1))
    axs[0,1].grid(True)
    axs[0,1].set_ylabel("Count of Facets", fontsize=12)
    axs[0,1].tick_params(labelsize=12)
    
    # MAE
    n3,_,_ = axs[1,0].hist(mae*1000, bins=np.arange(0,10,0.5), 
                edgecolor="black", facecolor="rebeccapurple")
    axs[1,0].set_title("Mean Absolute Error (" + units + ")", fontsize=14, weight='bold')
    axs[1,0].set(xlim=(0, 10))
    axs[1,0].grid(True)
    axs[1,0].set_ylabel("Count of Facets", fontsize=12)
    axs[1,0].tick_params(labelsize=12)
    
    # ME
    n4,_,_ = axs[1,1].hist(me*1000, bins=np.arange(-1.5,1.5,0.15), 
                edgecolor="black", facecolor="cornflowerblue")
    axs[1,1].set_title("Mean Error (" + units + ")", fontsize=14, weight='bold')
    axs[1,1].set(xlim=(-1.5, 1.5))
    axs[1,1].grid(True)
    axs[1,1].set_ylabel("Count of Facets", fontsize=12)
    axs[1,1].tick_params(labelsize=12)
    
    # set y limit
    ymax = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4)])
    axs[0,0].set(ylim=(0, ymax+1))
    axs[0,1].set(ylim=(0, ymax+1))
    axs[1,0].set(ylim=(0, ymax+1))
    axs[1,1].set(ylim=(0, ymax+1))
    
    # Save figure
    # plt.savefig(save_dir + str(name) + "_r2_slope_mae_mse.png", dpi=200, 
    #             transparent=True, bbox_inches='tight')
    
    # Show Figure
    plt.show()
    
    # Close Figure
    plt.close()
    
    
    
#%% 
""" FUNCTION DEFINITION: heatmap_by_facet
    INPUTS
    save_dir    - Directory to Save the Figure
    model       - Fitted Convolutional Neural Network
    opg_type    - What Type of OPG Values Was Used to Formulate Raw Values
    name        - Name of the Predicted Data
    atmos       - Xarray containing Atmospheric data
    opg         - Xarray containing OPG data
    
    OUTPUT - Heatmap Plot of True OPG Values VS Predicted OPG Values by Facet
"""

def heatmap_by_facet(save_dir, model, opg_type, name, atmos, opg):
    
    # create path to save
    path = save_dir + "heatmap_by_facet/"
    if os.path.exists(path) == False: os.mkdir(path)
    
    actual, predicted = formulate_actual_predicted(model, atmos, opg, opg_type)
    
    units = "mm/m"
    
    # pull range of opg values
    max_val = np.max((np.max(predicted[:]), np.max(actual[:])))
    min_val = np.min((np.min(predicted[:]), np.min(actual[:])))
    
    for facetx in range(len(opg.facet_num)):
        # pull opg num
        facet_num = opg.facet_num.values[facetx]
        
        label_size = 16
        tick_size = 14
        
        plt.figure(figsize=(8, 8))
        
        idx = actual[:,facetx] != 0.0
        
        heatmap, xedges, yedges = np.histogram2d(actual[idx,facetx], 
                                                 predicted[idx,facetx], 
                                                 bins=100,
                                                 range=[[min_val, max_val],[min_val, max_val]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        CMAP = ncm.cmap("MPL_gnuplot2")
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=CMAP, norm=colors.LogNorm())
        cb = plt.colorbar()
        plt.clim(10**0, 10**3)
    
        cb.ax.tick_params(labelsize=tick_size)
    
        plt.xlim((min_val, max_val))
        plt.ylim((min_val, max_val))
        
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        
        plt.plot([min_val, max_val],[min_val, max_val], color='r')    
        
        #find line of best fit
        m, b = np.polyfit(actual[idx,facetx], predicted[idx,facetx], 1)
        plt.plot([min_val, max_val], [m*min_val+b, m*max_val+b], color='dimgrey', linestyle=':')    
        
        plt.xlabel("Actual", fontsize=label_size)
        plt.ylabel("Predicted", fontsize=label_size)
        
        r2 = np.around(np.corrcoef(
                    actual[idx,facetx], predicted[idx,facetx])**2, decimals=2)
        # r_rank = np.around(spstats.spearmanr(
        #             actual[idx,facetx], predicted[idx,facetx]), decimals=2)
        mae = np.around(np.nanmean(np.abs(
                    actual[idx,facetx] - predicted[idx,facetx])), decimals=5)
        m = np.around(m, decimals=2)
        mse = np.around(np.nanmean((actual[idx,facetx] - predicted[idx,facetx])**2), decimals=6)
        
        plt.suptitle("Facet-" + str(facet_num) + " " +  name + " OPG Values (" + 
                  units + ")", fontweight="bold", y=0.88, x=0.44, fontsize=label_size)
        
        plt.title("$\mathregular{r^2}$: " + str(r2[0,1]) +  
                  ", MAE: " + str(mae) + 
                  ", MSE: " + str(mse) + 
                  ", Slope: " + str(m), fontsize=tick_size)
        
        plt.grid(True)
        
        # save figure
        # plt.savefig(path + name + '_actVSpred_heatmap_facet' + str(facet_num) + '.png', 
        #             dpi=200, transparent=True, bbox_inches='tight')
    
        plt.close()
        plt.show()
    
    
    
    
#%% 
""" FUNCTION DEFINITION: heatmap
    INPUTS
    save_dir    - Directory to Save the Figure
    model       - Fitted Convolutional Neural Network
    opg_type    - What Type of OPG Values Was Used to Formulate Raw Values
    name        - Name of the Predicted Data
    atmos       - Xarray containing Atmospheric data
    opg         - Xarray containing OPG data
    
    OUTPUT - Heatmap Plot of the True OPG Values VS the Predicted OPG Values
"""

def heatmap(save_dir, model, opg_type, name, atmos, opg):
    
    actual, predicted = formulate_actual_predicted(model, atmos, opg, opg_type)
    
    units = "mm/m"
    
    actual    = np.reshape(actual, -1)
    predicted = np.reshape(predicted, -1)
    idx       = actual != 0.0
    actual    = actual[idx]
    predicted = predicted[idx]
    
    max_val = np.max((np.max(predicted[:]), np.max(actual[:])))
    min_val = np.min((np.min(predicted[:]), np.min(actual[:])))
    
    label_size = 16
    tick_size = 14
    
    plt.figure(figsize=(8, 8))
    
    heatmap, xedges, yedges = np.histogram2d(np.reshape(actual, -1), np.reshape(predicted, -1), bins=100,
                                 range=[[min_val, max_val],[min_val, max_val]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    CMAP = ncm.cmap("MPL_gnuplot2")
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=CMAP, norm=colors.LogNorm())
    cb = plt.colorbar()

    cb.ax.tick_params(labelsize=tick_size)

    plt.xlim((min_val, max_val))
    plt.ylim((min_val, max_val))
    
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    
    plt.plot([min_val, max_val],[min_val, max_val], color='r')    
    
    #find line of best fit
    m, b = np.polyfit(np.reshape(actual, -1), np.reshape(predicted, -1), 1)
    plt.plot(np.reshape(actual, -1), m*np.reshape(actual, -1)+b, color='dimgrey', linestyle='--')
    
    plt.xlabel("Actual", fontsize=label_size)
    plt.ylabel("Predicted", fontsize=label_size)
    
    r2 = np.around(np.corrcoef(actual, predicted)**2, decimals=2)
    # r_rank = np.around(spstats.spearmanr(actual, predicted), decimals=2)
    mae = np.around(np.nanmean(np.abs(actual - predicted)), decimals=5)
    m = np.around(m, decimals=2)
    mse = np.around(np.nanmean((actual - predicted)**2), decimals=6)
    
    plt.suptitle(name + " OPG Values (" + units + ")", fontweight="bold", y=0.88, x=0.44,
                 fontsize=label_size)
    
    plt.title("$\mathregular{r^2}$: " + str(r2[0,1]) +  
              ", MAE: " + str(mae) + 
              ", MSE: " + str(mse) + 
              ", Slope: " + str(m), fontsize=tick_size)
    
    plt.grid(True)
    
    # save figure
    # plt.savefig(save_dir + name + '_actVSpred_heatmap.png', dpi=200, 
    #             transparent=True, bbox_inches='tight')
    
    plt.close()
    plt.show()




#%% 
""" FUNCTION DEFINITION: training_loss
    INPUTS
    save_dir        - Directory to Save the Figure
    history         - History Output From Training the CNN
    loss_metric     - Metric Used to Measure Loss
    opg_type    - What Type of OPG Values Was Used to Formulate Original Values
    
    OUTPUT - Lineplot of the Loss Values from Training the CNN
"""

def training_loss(save_dir, history, loss_metric, opg_type):
    
    # opg type presets
    if opg_type == 0: units = " (mm/m)"
    elif opg_type == 1: units = " (Standardized OPG)"
    elif opg_type == 2: units = " (Normalized OPG)"
    
    loss = history["loss"]
    epochs = range(1, len(loss)+1)
    
    plt.figure()
    
    plt.plot(epochs, loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel(loss_metric + units)
    plt.title("Training Loss")
    
    plt.legend()
    plt.grid(True)
    
    plt.close()
    plt.show()
    
    #plt.savefig("test_fig.png", dpi=300, transparent=True)

    
    
#%% 
""" FUNCTION DEFINITION: training_validation_loss
    INPUTS
    save_dir        - Directory to Save the Figure
    history         - History Output From Training the CNN
    loss_metric     - Metric Used to Measure Loss
    opg_type    - What Type of OPG Values Was Used to Formulate Original Values
        
    OUTPUT - Plot the Training and Validation Loss from Training the CNN
"""

def training_validation_loss(save_dir, history, loss_metric, opg_type):
    
    # opg type presets
    if opg_type == 0: units = " (mm/m)"
    elif opg_type == 1: units = " (Standardized OPG)"
    elif opg_type == 2: units = " (Normalized OPG)"
    
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(loss)+1)
    epoch_num= max(epochs)
    
    plt.figure(figsize=(10,5))
    
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel(loss_metric + units)
    
    plt.xlim([0,epoch_num])
    
    plt.legend()
    plt.grid(True)
    
    # Save figure
    # plt.savefig(save_dir + 'train_vldtn_loss.png', dpi=300, transparent=True, 
    #             bbox_inches='tight')

    plt.close()
    plt.show()



