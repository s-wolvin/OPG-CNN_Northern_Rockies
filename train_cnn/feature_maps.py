""" 
Savanna Wolvin
Created: Jun 1st, 2023
Edited: Jun 1st, 2023
    

##### SUMMARY #####
Script file holding the functions to evaluate input variables for the 
Convolutional Neural Network for cnn_regional_facet_MAIN.py

##### FUNCTION LIST ##########################################################
    plot_feature_maps_conv2d_1() - Function to plot feature maps from the input to the 
                            CNN *** Unfinished


"""
# %% Global Imports

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import os



     
# %%
""" FUNCTION DEFINITION: plot_feature_maps_conv2d_1
    INPUTS
    save_dir    - Directory to Save the Figures
    model       - Fitted ConvNet
    opg_type    - Type of OPG Value Used in Training
    name        - If These are Training, Testing, or Validation Dates
    atmos       - Xarray Contianing the Atmospheric Values Used
    opg         - Xarray of OPG Values Used
    
    OUTPUT - Plot Feature Maps
"""


def plot_feature_maps_conv2d_1(save_dir, model, opg_type, name, atmos, opg):
    
    # Calculate Size of Data
    atm_sz = dict(atmos.sizes)
    opg_sz = dict(opg.sizes)
    atmos_names = np.array(atmos.data_vars)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos.values():
        if len(ii.dims) == 3:
            count_channels += 1
        elif len(ii.dims) == 2:
            count_on_facet += 1
            atmos_names = np.delete(
                atmos_names, np.argmax(ii.name == atmos_names))
    
    # pull list of OPG values
    opg_values = np.array(opg['opg'])
    
    # pull plotting axis values
    lons = np.array(atmos['lon'])
    lats = np.array(atmos['lat'])
    date = atmos['time']
    loni, lati = np.meshgrid(lons, lats)
    extent = [-150, -100, 25, 57.5]
    cmap_feat = 'viridis' #ncm.cmap('MPL_viridis')
    
    # loop through each day
    for dayx in range(0, atm_sz['time']):
        if np.sum(np.abs(opg_values[dayx, :] > 0)) > 0:
            # pull values from that day
            # Create Empty Array for Testing Data
            atmos_test_4D_X = np.zeros(
                (1, atm_sz['lat'], atm_sz['lon'], count_channels))
            atmos_test_OF_X = np.zeros((1, opg_sz['facet_num']*count_on_facet))

            # Pull Variables for Testing
            count_4D = 0
            count_OF = 0
            for var_name, values in atmos.items():
                if len(values.dims) == 3:
                    atmos_test_4D_X[:, :, :, count_4D] = np.array(
                        values[dayx, :, :])
                    count_4D += 1
                elif len(values.dims) == 2:
                    atmos_test_OF_X[:, opg_sz['facet_num']*count_OF:opg_sz['facet_num']
                                    * (count_OF+1)] = np.array(values[dayx, :])
                    count_OF += 1

            # Combine Inputs if On-Facet Data Exists
            if count_on_facet > 0:
                inputs = [atmos_test_4D_X, atmos_test_OF_X]
            else:
                inputs = [atmos_test_4D_X]
    
            # Create a model that takes the input image and outputs the first
            # convolutional layer
            feat_model = tf.keras.models.Model([model.inputs], [model.get_layer('conv2d_1').output])
        
            # Then compute the gradient of the top predicted class for our
            # input image with resepct to the activations of the last conv layer
            first_conv = feat_model(inputs)
    
            ##### PLOT THE FEATURE MAPS ######################################
            # find size of figure
            x_axis = 4
            y_axis = 8 
            
            # Plot Transforms
            datacrs = ccrs.PlateCarree()
            projex = ccrs.Mercator(central_longitude=np.mean(lons))
            
            # new figure like michaels
            fig, ax = plt.subplots(nrows=y_axis, ncols=x_axis,
                                   figsize=(x_axis*4, y_axis*4),
                                   subplot_kw={'projection': projex})
            
            ax_count = 0
            for row in range(y_axis):
                for col in range(x_axis):
                    if ax_count < np.shape(first_conv)[3]:
                        # Add Feature Map
                        ax[row, col].contourf(loni, lati, np.squeeze(
                                        first_conv[:, :, :, ax_count]),
                                        zorder=1, cmap=cmap_feat, transform=datacrs)
    
                        # Cartography Features
                        states_provinces = cfeat.NaturalEarthFeature(
                                                category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')
                        ax[row, col].add_feature(cfeat.COASTLINE.with_scale(
                            '110m'), zorder=2, edgecolor="white")
                        ax[row, col].add_feature(cfeat.BORDERS.with_scale(
                            '110m'), zorder=3, edgecolor="white")
                        ax[row, col].add_feature(states_provinces, 
                                     zorder=4, edgecolor="white")
                        ax[row, col].set_extent(extent)
                        
                        ax_count += 1
            
            plt.subplots_adjust(wspace=0, hspace=0)
            
            # Add Title with Date and Actual/Predicted Values
            df = np.array(date[dayx], dtype='str')
            df = str(df)
            plt.suptitle("Convolution 1: "+df[0:10], fontsize=36, y=0.92, weight='bold')
            
            plt.show()
            
            # create path to save
            path = save_dir + "feat_map/"
            if os.path.exists(path) == False:
                os.mkdir(path)

            plt.savefig(path + df[0:10] + "_conv2d_1_" + name + ".png", dpi=400, transparent=True,
                        bbox_inches='tight')
            
            plt.close()




# %%
""" FUNCTION DEFINITION: plot_feature_maps_conv2d_2
    INPUTS
    save_dir    - Directory to Save the Figures
    model       - Fitted ConvNet
    opg_type    - Type of OPG Value Used in Training
    name        - If These are Training, Testing, or Validation Dates
    atmos       - Xarray Contianing the Atmospheric Values Used
    opg         - Xarray of OPG Values Used
    
    OUTPUT - Plot Feature Maps
"""


def plot_feature_maps_conv2d_2(save_dir, model, opg_type, name, atmos, opg):
    
    # Calculate Size of Data
    atm_sz = dict(atmos.sizes)
    opg_sz = dict(opg.sizes)
    atmos_names = np.array(atmos.data_vars)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos.values():
        if len(ii.dims) == 3:
            count_channels += 1
        elif len(ii.dims) == 2:
            count_on_facet += 1
            atmos_names = np.delete(
                atmos_names, np.argmax(ii.name == atmos_names))
    
    # pull list of OPG values
    opg_values = np.array(opg['opg'])
    
    # pull plotting axis values
    lats = np.arange(np.min(atmos['lat'])+0.5, np.max(atmos['lat'])+0.5, 1)
    lons = np.arange(np.min(atmos['lon'])+0.5, np.max(atmos['lon'])+0.5, 1)
    date = atmos['time']
    loni, lati = np.meshgrid(lons, lats)
    extent = [-150, -100, 25, 57.5]
    cmap_feat = 'viridis' #ncm.cmap('MPL_viridis')
    
    # loop through each day
    for dayx in range(0, atm_sz['time']):
        if np.sum(np.abs(opg_values[dayx, :] > 0)) > 0:
            # pull values from that day
            # Create Empty Array for Testing Data
            atmos_test_4D_X = np.zeros(
                (1, atm_sz['lat'], atm_sz['lon'], count_channels))
            atmos_test_OF_X = np.zeros((1, opg_sz['facet_num']*count_on_facet))

            # Pull Variables for Testing
            count_4D = 0
            count_OF = 0
            for var_name, values in atmos.items():
                if len(values.dims) == 3:
                    atmos_test_4D_X[:, :, :, count_4D] = np.array(
                        values[dayx, :, :])
                    count_4D += 1
                elif len(values.dims) == 2:
                    atmos_test_OF_X[:, opg_sz['facet_num']*count_OF:opg_sz['facet_num']
                                    * (count_OF+1)] = np.array(values[dayx, :])
                    count_OF += 1

            # Combine Inputs if On-Facet Data Exists
            if count_on_facet > 0:
                inputs = [atmos_test_4D_X, atmos_test_OF_X]
            else:
                inputs = [atmos_test_4D_X]
    
            # Create a model that takes the input image and outputs the first
            # convolutional layer
            feat_model = tf.keras.models.Model([model.inputs], [model.get_layer('conv2d_2').output])
        
            # Then compute the gradient of the top predicted class for our
            # input image with resepct to the activations of the last conv layer
            first_conv = feat_model(inputs)
    
            ##### PLOT THE FEATURE MAPS ######################################
            # find size of figure
            x_axis = 4 #np.round(np.sqrt(np.shape(first_conv)[3]+1)).astype('int')
            y_axis = 16 #np.ceil(np.sqrt(np.shape(first_conv)[3]+1)).astype('int')
            
            # Plot Transforms
            datacrs = ccrs.PlateCarree()
            projex = ccrs.Mercator(central_longitude=np.mean(lons))
            
            # new figure like michaels
            fig, ax = plt.subplots(nrows=y_axis, ncols=x_axis,
                                   figsize=(x_axis*4, y_axis*4),
                                   subplot_kw={'projection': projex})
            
            ax_count = 0
            for row in range(y_axis):
                for col in range(x_axis):
                    if ax_count < np.shape(first_conv)[3]:
                        # Add Feature Map
                        ax[row, col].contourf(loni, lati, np.squeeze(
                                        first_conv[:, :, :, ax_count]),
                                        zorder=1, cmap=cmap_feat, transform=datacrs)
    
                        # Cartography Features
                        states_provinces = cfeat.NaturalEarthFeature(
                                                category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')
                        ax[row, col].add_feature(cfeat.COASTLINE.with_scale(
                            '110m'), zorder=2, edgecolor="white")
                        ax[row, col].add_feature(cfeat.BORDERS.with_scale(
                            '110m'), zorder=3, edgecolor="white")
                        ax[row, col].add_feature(states_provinces, 
                                     zorder=4, edgecolor="white")
                        ax[row, col].set_extent(extent)
                        
                        ax_count += 1
            
            plt.subplots_adjust(wspace=0, hspace=0)
            
            # Add Title with Date and Actual/Predicted Values
            df = np.array(date[dayx], dtype='str')
            df = str(df)
            plt.suptitle("Convolution 2: "+df[0:10], fontsize=54, y=0.91, weight='bold')
            
            plt.show()
            
            # create path to save
            path = save_dir + "feat_map/"
            if os.path.exists(path) == False:
                os.mkdir(path)

            plt.savefig(path + df[0:10] + "_conv2d_2_" + name + ".png", dpi=400, transparent=True,
                        bbox_inches='tight')
            
            plt.close()
            
            
            
            