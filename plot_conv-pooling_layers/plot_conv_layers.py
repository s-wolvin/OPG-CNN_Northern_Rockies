"""
Savanna Wolvin
Created: Jun 5th, 2023
Edited: Jun 5th, 2023


##### Summary ################################################################
This script loads in the model and datasets and plots the convolutional layers.


##### Input ###################################################################



##### Output ##################################################################


"""
#%% Global Imports

import tensorflow as tf
import tensorflow.keras as tf_k
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import nclcmaps as ncm
import os




#%% Variable Presets

model_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_facet_cnn_weighting/NR/"
model_name = "2024-01-08_1113"

atmos_subset = "training" # training, testing, validation
atmos_name = ["IVT", "Geo. Hgt 500 hPa", "Accum. Precip.", 
              "SH 850 hPa", "Temp. 700 hPa", "U-Wind 700 hPa",
              "V-Wind 10-meter", "W-Wind 700 hPa"]

save_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/NR/"


#%% Figure Presets

extent = [-150, -100, 25, 57.5]
#cmap_feat = 'viridis'
cmap_feat = ncm.cmap('NCV_blu_red')




#%% Custom Object

class weighted_MSE(tf_k.losses.Loss):
    def call(self, y_true, y_pred):
        se  = tf.math.square(y_pred - y_true)
        mask = tf.cast(y_true != 0, tf.float32)
        weight_se = se * mask
        weight_mse = tf.reduce_mean(weight_se, axis=-1)
        return weight_mse
    
    


#%% Load Data

print("Load CNN...")

# Load Model
model = tf_k.models.load_model(f"{model_dir}{model_name}/CNN-OPG_MODEL", 
                                   custom_objects={'weighted_MSE': weighted_MSE})

# Load Atmos Data
opg = xr.open_dataset(f"{model_dir}{model_name}/datasets/opg_{atmos_subset}.nc", engine='netcdf4')
atmos = xr.open_dataset(f"{model_dir}{model_name}/datasets/atmos_{atmos_subset}.nc", engine='netcdf4')




#%% Pull Atmos Data

print("Load Atmos Data...")

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




#%% Pull Layer Names

print("Pull layer names...")

layers = [layer.name for layer in model.layers]
layers = [lx for lx in layers if lx[0:3] == "max" or lx[0:4] == "conv" or lx[0:5]=='input']
date = atmos['time']
opg_values = np.array(opg['opg'])





#%% Loop through each layer and plot output

print("Plotting...")

for dayx in range(0, atm_sz['time']):
    if np.sum(np.abs(opg_values[dayx, :]) > 0) > 50:
        for lx in layers:
                
            # Combine Inputs if On-Facet Data Exists
            if count_on_facet > 0:
                inputs = [atmos_train_4D_X[dayx,:,:,:], atmos_train_OF_X[dayx,:,:]]
            else:
                inputs = [np.expand_dims(atmos_train_4D_X[dayx,:,:,:], axis=0)]
            
            # Create a model that takes the input image and outputs the first
            # convolutional layer
            feat_model = tf.keras.models.Model([model.inputs], [model.get_layer(lx).output])
            
            # Then compute the gradient of the top predicted class for our
            # input image with resepct to the activations of the layer
            LAYER = feat_model(inputs)
            
            # Plot presets
            if lx == 'input':
                x_axis = 1
                spacing = 0.01
                dpix = 200
            else:
                LAYER = (LAYER - np.mean(LAYER)) / np.std(LAYER)
                spacing = 0.01
                if lx[-1] == "1": 
                    x_axis = 2
                    dpix = 150
                elif lx[-1] == "2": 
                    x_axis = 3
                    dpix = 90
                elif lx[-1] == "3": 
                    x_axis = 4
                    dpix = 50
            
            # pull plotting axis values
            lats = np.linspace(np.min(atmos['lat']), np.max(atmos['lat']), 
                               num=np.shape(LAYER)[1], endpoint=True)
            lons = np.linspace(np.min(atmos['lon']), np.max(atmos['lon']), 
                               num=np.shape(LAYER)[2], endpoint=True)
            loni, lati = np.meshgrid(lons, lats)
            
            # find size of figure
            y_axis = np.ceil(np.shape(LAYER)[3]/x_axis).astype('int')
            
            # Plot Transforms
            datacrs = ccrs.PlateCarree()
            projex = ccrs.Mercator(central_longitude=np.mean(lons))
            
            # new figure like michaels
            fig, ax = plt.subplots(nrows=y_axis, ncols=x_axis,
                                   figsize=(x_axis*4, y_axis*4),
                                   subplot_kw={'projection': projex})
            
            if lx == 'input':
                ax_count = 0
                for row in range(y_axis):
                    if ax_count < np.shape(LAYER)[3]:
                        # Add Feature Map                    
                        ax[row].pcolor(loni, lati, np.squeeze(
                                        LAYER[:, :, :, ax_count]), clim=[-6, 6],
                                        zorder=1, cmap=cmap_feat, transform=datacrs)
    
                        # Cartography Features
                        states_provinces = cfeat.NaturalEarthFeature(
                                                category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')
                        ax[row].add_feature(cfeat.COASTLINE.with_scale(
                            '110m'), zorder=2, edgecolor="black")
                        ax[row].add_feature(cfeat.BORDERS.with_scale(
                            '110m'), zorder=3, edgecolor="black")
                        ax[row].add_feature(states_provinces, 
                                     zorder=4, edgecolor="black")
                        ax[row].set_extent(extent)
                        
                        # if lx == "input":
                        #     ax[row, col].set_title(atmos_name[ax_count], fontsize=22,
                        #                            weight='bold')
                        
                        ax_count += 1
                        
                    elif ax_count >= np.shape(LAYER)[3]:
                        
                        ax[row].axis('off')
            
                plt.subplots_adjust(wspace=0.01, hspace=spacing)
                
            else: 
                ax_count = 0
                for row in range(y_axis):
                    for col in range(x_axis):
                        if ax_count < np.shape(LAYER)[3]:
                            # Add Feature Map                    
                            ax[row, col].pcolor(loni, lati, np.squeeze(
                                            LAYER[:, :, :, ax_count]), clim=[-6, 6],
                                            zorder=1, cmap=cmap_feat, transform=datacrs)
        
                            # Cartography Features
                            states_provinces = cfeat.NaturalEarthFeature(
                                                    category='cultural',
                                                    name='admin_1_states_provinces_lines',
                                                    scale='50m', facecolor='none')
                            ax[row, col].add_feature(cfeat.COASTLINE.with_scale(
                                '110m'), zorder=2, edgecolor="black")
                            ax[row, col].add_feature(cfeat.BORDERS.with_scale(
                                '110m'), zorder=3, edgecolor="black")
                            ax[row, col].add_feature(states_provinces, 
                                         zorder=4, edgecolor="black")
                            ax[row, col].set_extent(extent)
                            
                            # if lx == "input":
                            #     ax[row, col].set_title(atmos_name[ax_count], fontsize=22,
                            #                            weight='bold')
                            
                            ax_count += 1
                            
                        elif ax_count >= np.shape(LAYER)[3]:
                            
                            ax[row, col].axis('off')
                
                plt.subplots_adjust(wspace=0.01, hspace=spacing)
            
            # Add Title with Date and Actual/Predicted Values
            df = np.array(date[dayx], dtype='str')
            df = str(df)
            #plt.suptitle(df[0:10], fontsize=36, y=0.92, weight='bold')
            
            # create path to save
            path = model_dir + model_name + "/cnn_layers/"
            if os.path.exists(path) == False:
                os.mkdir(path)
    
            plt.savefig(f"{path}{df[0:10]}_{atmos_subset}_{lx}.png", dpi=dpix, 
                        transparent=True, bbox_inches='tight')
            
            # plt.show()
            
            plt.close()
            
                
                
                
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    



























