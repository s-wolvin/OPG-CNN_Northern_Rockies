"""
Savanna Wolvin
Created: Aug 8th, 2023
Edited: Aug 8th, 2025


##### Summary ################################################################
This script plots composite Gradient-weighted Class Activation Maps (Grad-CAMs)
based on the k-means clusters of OPG event types. It plots either Grad-CAMs of 
the training or testing subset. The variables it plots are as what the CNN was 
trained on: 
    500 hPa geopotential height
    700 hPa u-winds
    IVT
    700 hPa temperature
    10-m v-winds
    700 hPa vertical velocity
    850 hPa specific humidity.

However, the plots use the total winds at 700 hPa and at 10-m. In addition, the 
CNN is trained on Z-Scored variables, here we plot the true values of IVT, 
700 hPa winds, precipitation, and 10-m winds. Grad-CAMs are indicated using 
stippling over the atmospheric variables.

Additionally, daily Grad-CAMs are plotted after the clusters.

##### Input ##################################################################
model_dir       - Directory to the CNN
model_name      - Folder of the CNN
atmos_subset    - Subset of the Atmospheric data to use
kmeans_dir      - Directory to the K-Means analysis of OPG events



"""
#%% Global Imports

import tensorflow.keras as tf_k
import tensorflow as tf
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeat
import os
import sys
sys.path.insert(1, '/uufs/chpc.utah.edu/common/home/u1324060/nclcmappy/')
import nclcmaps as ncm




#%% Preset Variables

model_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_facet_cnn_weighting/NR/"
model_name = "2024-01-08_1113"

atmos_subset = "testing"

kmeans_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_kmeans_clust/"



#%% Custom Object

class weighted_MSE(tf_k.losses.Loss):
    def call(self, y_true, y_pred):
        se  = tf.math.square(y_pred - y_true)
        mask = tf.cast(y_true != 0, tf.float32)
        weight_se = se * mask
        weight_mse = tf.reduce_mean(weight_se, axis=-1)
        return weight_mse
    
    


#%% Load Data

# Load Model
model = tf_k.models.load_model(f"{model_dir}{model_name}/CNN-OPG_MODEL", 
                                   custom_objects={'weighted_MSE': weighted_MSE})

# Load Atmos Data
opg = xr.open_dataset(f"{model_dir}{model_name}/datasets/opg_{atmos_subset}.nc", engine='netcdf4')
atmos = xr.open_dataset(f"{model_dir}{model_name}/datasets/atmos_{atmos_subset}.nc", engine='netcdf4')
atmos_mean = xr.open_dataset(f"{model_dir}{model_name}/datasets/atmos_mean.nc", engine='netcdf4')
atmos_std = xr.open_dataset(f"{model_dir}{model_name}/datasets/atmos_standardDeviation.nc", engine='netcdf4')

# Load Kmeans Data
kmeans = pd.read_csv(f"{kmeans_dir}kmeans_clusters_opg_NR_5_clusters")


#%% Plot Presets 

x_axis = 3
cmap_feat = ncm.cmap('NCV_blu_red')
extent = [-150, -100, 25, 57.5]
levs = np.arange(-2, 2.1, 0.1)

# load regional location
rgn_bnds = opg.attrs['bounds']
rgn_lons = [rgn_bnds[2], rgn_bnds[2],
            rgn_bnds[3], rgn_bnds[3], rgn_bnds[2]]
rgn_lats = [rgn_bnds[0], rgn_bnds[1],
            rgn_bnds[1], rgn_bnds[0], rgn_bnds[0]]



#%% Select Atmos Data

def atmos_cluster(clust):
    
    # Pull Dates with that cluster
    clust_dates = kmeans['datetime'][kmeans['cluster']==clust]
    clust_dates = pd.to_datetime(clust_dates)

    # Pull Dates within the Cluster
    atmos_time = pd.to_datetime(atmos.time.values)
    atmos_time = atmos_time.drop(atmos_time.difference(clust_dates))
    
    atmos_clust_mean = atmos.sel(time=atmos_time).mean('time')
    
    # atmos_clust_mean.IVTsfc = (atmos_clust_mean.IVTsfc * atmos_std.IVTsfc) + atmos_mean.IVTsfc
    
    atmos_clust_mean = (atmos_clust_mean * atmos_std) + atmos_mean
    
    # Additionally pull other winds
    v700 = xr.Dataset()
    u10m = xr.Dataset()

    for yr in np.unique(atmos_time.year):
        v700x = xr.open_dataset(f"/uufs/chpc.utah.edu/common/home/strong-group7/savanna/ecmwf_era5/western_CONUS/daily/press/era5_vwnd_{yr}_oct-apr_daily.nc")
        v700x = v700x.sel(level = 700)
        v700 = xr.merge([v700, v700x])
    
        u10mx = xr.open_dataset(f"/uufs/chpc.utah.edu/common/home/strong-group7/savanna/ecmwf_era5/western_CONUS/daily/sfc/era5_uwnd_10m_{yr}_oct-apr_daily.nc")
        u10m = xr.merge([u10m, u10mx])
    
    v700 = v700.sel(time = atmos_time, latitude = atmos.lat.values, longitude = atmos.lon.values)
    u10m = u10m.sel(time = atmos_time, latitude = atmos.lat.values, longitude = atmos.lon.values)
    
    v700 = v700.rename({'vwnd': 'vwnd700', 'latitude': 'lat','longitude': 'lon'})
    u10m = u10m.rename({'uwnd_10m': 'uwnd_10msfc', 'latitude': 'lat','longitude': 'lon'})
    atmos2 = xr.merge([v700, u10m])
    
    atmosx = atmos.sel(time=atmos_time)
    
    # Pull selected Atmos Dates and Take Time Mean
    # atmos_clust_mean = (atmos * atmos_std) + atmos_mean
    atmos_clust_mean = xr.merge([atmosx, atmos2])
    
    
    atmos_clust_mean = atmosx.mean('time')
    
    # atmos_clust_mean.IVTsfc = (atmos_clust_mean.IVTsfc * atmos_std.IVTsfc) + atmos_mean.IVTsfc
    
    atmos_clust_mean = (atmos_clust_mean * atmos_std) + atmos_mean
    
    atmos_clust_mean = xr.merge([atmos_clust_mean, atmos2.mean('time')])

    return atmos_clust_mean, atmosx, atmos_time, atmos2




#%% 

def grad_cam(atmos_clust):
    
    # needed vars
    opg_sz = dict(opg.sizes)
    
    
    # Calculate Size of Data
    atm_sz = dict(atmos_clust.sizes)
    opg_sz = dict(opg.sizes)
    atmos_names = np.array(atmos_clust.data_vars)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos_clust.values():
        if len(ii.dims) == 3:
            count_channels += 1
        elif len(ii.dims) == 2:
            count_on_facet += 1
            atmos_names = np.delete(
                atmos_names, np.argmax(ii.name == atmos_names))
    
    
    # Create Empty Array for Testing Data
    atmos_test_4D_X = np.zeros(
        (atm_sz['time'], atm_sz['lat'], atm_sz['lon'], count_channels))
    atmos_test_OF_X = np.zeros((1, opg_sz['facet_num']*count_on_facet))

    # Pull Variables for Testing
    count_4D = 0
    count_OF = 0
    for var_name, values in atmos_clust.items():
        if len(values.dims) == 3:
            atmos_test_4D_X[:, :, :, count_4D] = np.array(
                values[:, :, :])
            count_4D += 1
        elif len(values.dims) == 2:
            atmos_test_OF_X[:, opg_sz['facet_num']*count_OF:opg_sz['facet_num']
                            * (count_OF+1)] = np.array(values[:, :])
            count_OF += 1

    # Combine Inputs if On-Facet Data Exists
    if count_on_facet > 0:
        inputs = [atmos_test_4D_X, atmos_test_OF_X]
    else:
        inputs = [atmos_test_4D_X]
    
    # Calculate the conposite grad-cam
    heatmap = []

    for output_node in range(opg_sz['facet_num']):
        # Create a model that takes the input image and outputs the last
        # convolutional layer and the output predictions
        grad_model = tf.keras.models.Model([model.inputs],
                                           [model.get_layer('conv2d_3').output, model.output])

        # Then compute the gradient of the top predicted class for our
        # input image with resepct to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(inputs)
            class_channel = preds[:, output_node]

        # This is the gradient of the output neuron with regard to the
        # output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Formulates the mean gradient of each feature map, how important is each feature map?
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # Reduce to 3 dimentions
        #last_conv_layer_output = last_conv_layer_output[0]

        # Multiply each channel of the feature map by how important it is
        heatmapx = np.multiply(last_conv_layer_output, 
                               pooled_grads[:, tf.newaxis, tf.newaxis, :])
        
        heatmapx = np.sum(heatmapx, axis = 3)
        
        #heatmapx = last_conv_layer_output[0,...] @ pooled_grads[0, tf.newaxis, tf.newaxis, :]

        # Reduce dimentions
        #heatmapx = np.mean(heatmapx, axis=0)
        heatmap.append(np.array(heatmapx[..., np.newaxis]))

    heatmap = np.concatenate(heatmap, axis=3)  # combine all maps
    heatmap = np.max(heatmap, axis=3) # create max of all between facets per day
    
    heatmap = np.mean(heatmap, axis=0) # create mean of all days

    heatmap = heatmap / np.max(heatmap[:])  # Scale to max of 1
    heatmap[heatmap < 0] = 0  # Remove all the negative values
    
    return heatmap



#%%############################################################################
# SAME PLOTS BUT ALL HORIZONTAL
###############################################################################
# PLot presets
x_axis = 3

def rename_title(var_name):
    
    name = var_name[:-3]
    level = var_name[-3:]
    
    # set begining of title
    if name == "IVT":       title = "IVT"
    elif name == "hgt":     title = "Geopotential" + "\n" + "Height"
    elif name == "precip":  title = "Accumulated" + "\n" + "Precipitation"
    elif name == "shum":    title = "Specific" + "\n" + "Humidity"
    elif name == "temp":    title = "Temperature" + "\n"
    elif name == "uwnd":    title = "U-Wind"
    elif name == "vwnd":    title = "V-Wind"
    elif name == "vwnd_10m": title = "V-Wind 10-m"
    elif name == "wwnd":    title = "W-Wind"
    
    # Set end of title
    if level != "sfc":
        title = title + " " + level + " hPa"
            
    return title


#%% plot presets

IVT_cmap = ncm.cmap('WhiteYellowOrangeRed')
IVT_levs = np.arange(100, 500, 50)

ANOM_levs = np.arange(-3, 3.25, 0.25)

uwnd_scl = 145
uwnd_col = [0.1, 0.1, 0.1]

pr_cmap = ncm.cmapDiscrete('prcp_2', indexList=np.arange(1,12))
pr_levs = np.arange(0.5, 6.5, 0.5)

temp_cmap = ncm.cmapDiscrete('BlRe', indexList=[24,24,24,24,24,24,24, 72,72,72,72,72,72])
temp_levs = np.arange(-32, 40, 4)

vwnd_scl = 60

wwnd_cmap = ncm.cmap('BlueDarkRed18')
wwnd_levs = np.array([0.5, 100])

shum_cmap = ncm.cmap('MPL_BrBG')
shum_levs = np.arange(-1.5, 1.75, 0.25)

border_color = [0.45, 0.45, 0.45]

apct = 14
pdng = 0.03
lbl_sz = 14
frc = 0.15

gradcam_cmap = ncm.cmap('MPL_Greys')



#%% Loop through each cluster and plot mean of atmosphere

for clust in np.unique(kmeans['cluster'].values):
    
    atmos_clust_mean, atmos_clust, atmos_time, atmos2 = atmos_cluster(clust)
    
    grad_cam_clust = grad_cam(atmos_clust)

    # pull plotting axis values
    loni, lati = np.meshgrid(atmos['lon'], atmos['lat'])
    
    # find size of figure
    atmos_name = [i for i in atmos_clust.data_vars]
    
    # interpolate the heatmap grid
    grad_cam_clust_size = np.shape(grad_cam_clust)
    heatmap_lat = np.linspace(
        np.min(atmos['lat']), np.max(atmos['lat']), num=grad_cam_clust_size[0])
    heatmap_lon = np.linspace(
        np.min(atmos['lon']), np.max(atmos['lon']), num=grad_cam_clust_size[1])
    h_lon, h_lat = np.meshgrid(heatmap_lon, heatmap_lat)
    
    # Plot Transforms
    datacrs = ccrs.PlateCarree()
    projex = ccrs.Mercator(central_longitude=np.mean(atmos['lon'].values))
    
    # Change vars
    # hgt500 = atmos_clust_mean.hgt500.values / 9.81
    
    hgt500ANOM = np.array(atmos_clust.hgt500.mean('time'))
    temp700ANOM = np.array(atmos_clust.temp700.mean('time'))
    shum850ANOM = np.array(atmos_clust.shum850.mean('time'))
    wwnd700ANOM = np.array(atmos_clust.wwnd700.mean('time'))
    
    IVTsfc = atmos_clust_mean.IVTsfc.values
    IVTsfc[IVTsfc < 100] = np.nan
    # temp700 = atmos_clust_mean.temp700.values - 273.15
    precipsfc = atmos_clust_mean.precipsfc.values * 1000
    precipsfc[precipsfc < 0.5] = np.nan
    shum850 = atmos_clust_mean.shum850.values * 1000
    shum850[shum850 < 0.1] = np.nan
    
    # new figure like michaels
    fig, ax = plt.subplots(nrows=1, ncols=x_axis,
                            figsize=(x_axis*4, 7),
                            subplot_kw={'projection': projex})
    
    ##### Plot 1: Geo Height, IVT, U-Winds
    # geo_lbl = ax[0].contour(loni, lati, hgt500, colors='black', 
    #               transform=datacrs, zorder=6)
    # ax[0].clabel(geo_lbl, fmt="%.0f", inline=True)
    
    geo_lbl = ax[0].contour(loni, lati, hgt500ANOM, levels=ANOM_levs,
                  cmap=temp_cmap, transform=datacrs, zorder=6, 
                  linestyles=np.where(ANOM_levs > 0, "-", "--"))
    ax[0].clabel(geo_lbl, fmt="%.2f", inline=True)
    
    
    IVT_ax = ax[0].contourf(loni, lati, IVTsfc, cmap=IVT_cmap, 
                    levels=IVT_levs, transform=datacrs, extend='max', zorder=2)
    # q = ax[0].quiver(loni[::10, ::15], lati[::10, ::15], atmos_clust_mean.uwnd700.values[::10, ::15], 
    #               np.zeros(np.shape(atmos_clust_mean.uwnd700.values))[::10, ::15], 
    #               scale=uwnd_scl, pivot='mid', transform=datacrs, zorder=7,
    #               color=uwnd_col, headwidth=6)
    q = ax[0].quiver(loni[::10, ::15], lati[::10, ::15], atmos_clust_mean.uwnd700.values[::10, ::15], 
                  atmos_clust_mean.vwnd700.values[::10, ::15], 
                  scale=uwnd_scl, pivot='mid', transform=datacrs, zorder=7,
                  color=uwnd_col, headwidth=6)
    ax[0].quiverkey(q, X=0.285, Y=0.7, U=15, label='15 m s$^{-1}$', labelpos='E', 
                    coordinates='figure', fontproperties={'size':lbl_sz})
    
    
    # Colorbar
    cb_IVT = plt.colorbar(IVT_ax, extend='max', pad=pdng, aspect=apct, fraction=frc, 
                          orientation='horizontal', ticks=[100,200,300,400])
    cb_IVT.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", size=lbl_sz)
    cb_IVT.ax.tick_params(labelsize=lbl_sz)
    
    # Cartography Features
    states_provinces = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')
    ax[0].add_feature(cfeat.LAND, facecolor='lightgray', zorder=1)
    ax[0].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=3, edgecolor=border_color)
    ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), zorder=4, edgecolor=border_color)
    ax[0].add_feature(states_provinces, zorder=5, edgecolor=border_color)
    ax[0].set_extent(extent)
    
    # hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
    #                         transform=datacrs, hatches=[None, '..', 'oo','OO'], 
    #                         levels=[0, 0.25, 0.50, 0.75, 1.00])
    
    if clust+1 == 1:
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, '..', '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 2:
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, '..', '..', '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 3:
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 4:
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 5:
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    else:
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    
    
    
    ax[0].set_title('500 hPa Hgt. Anom. (contour)' + '\n' + 
                    '700 hPa Winds (                     )', fontsize=lbl_sz)
    
    if atmos_subset == "training": 
        ax[0].text(-151, 42.5, 'Training', va='bottom', ha='center', 
                    rotation='vertical', rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz)
        ax[0].text(-147, 58, 'a)', va='center', ha='center', 
                    rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz, backgroundcolor='white', zorder=9)
    elif atmos_subset == "testing": 
        ax[0].text(-151, 42.5, 'Testing', va='bottom', ha='center', 
                    rotation='vertical', rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz)
        ax[0].text(-147, 58, 'd)', va='center', ha='center', 
                    rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz, backgroundcolor='white', zorder=9)
    
    ##### Plot 2: Precip, temp, v-winds
    # temp_lbl = ax[1].contour(loni, lati, temp700, levels=temp_levs,
    #               cmap=temp_cmap, transform=datacrs, zorder=6, 
    #               linestyles=np.where(temp_levs > 0, "-", "--"))
    # ax[1].clabel(temp_lbl, fmt="%.0f", inline=True, fontsize=lbl_sz)
    
    temp_lbl = ax[1].contour(loni, lati, temp700ANOM, levels=ANOM_levs,
                  cmap=temp_cmap, transform=datacrs, zorder=6, 
                  linestyles=np.where(ANOM_levs > 0, "-", "--"))
    ax[1].clabel(temp_lbl, fmt="%.2f", inline=True)  
    
    
    pr_ax = ax[1].contourf(loni, lati, precipsfc, cmap=pr_cmap, 
                    levels=pr_levs, transform=datacrs, extend='max', zorder=2)
    # q = ax[1].quiver(loni[::15, ::10], lati[::15, ::10], np.zeros(np.shape(atmos_clust_mean.vwnd_10msfc.values))[::15, ::10],
    #               atmos_clust_mean.vwnd_10msfc.values[::15, ::10], scale=vwnd_scl, 
    #               transform=datacrs, zorder=7, headwidth=6)
    q = ax[1].quiver(loni[::15, ::10], lati[::15, ::10], atmos_clust_mean.uwnd_10msfc.values[::15, ::10],
                  atmos_clust_mean.vwnd_10msfc.values[::15, ::10], scale=vwnd_scl, 
                  transform=datacrs, zorder=7, headwidth=6)
    
    ax[1].quiverkey(q, X=0.54, Y=0.7, U=5, label='5 m s$^{-1}$', labelpos='E', 
                    coordinates='figure', fontproperties={'size':lbl_sz})
    
    # Colorbar
    cb_pr = plt.colorbar(pr_ax, pad=pdng, aspect=apct, fraction=frc, 
                          orientation='horizontal', extend='max', 
                          ticks=[1,2,3,4,5,6])
    cb_pr.set_label("Precipitation (mm)", size=lbl_sz)
    cb_pr.ax.tick_params(labelsize=lbl_sz)
    
    # Cartography Features
    ax[1].add_feature(cfeat.LAND, facecolor='lightgray', zorder=1)
    ax[1].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=3, edgecolor=border_color)
    ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), zorder=4, edgecolor=border_color)
    ax[1].add_feature(states_provinces, zorder=5, edgecolor=border_color)
    ax[1].set_extent(extent)
    
    if clust+1 == 1:
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, '..', '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 2:
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, '..', '..', '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 3:
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 4:
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 5:
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    else:
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    
    ax[1].set_title('700 hPa Temp. Anom. (contour)' + '\n' + 
                    '10-m Winds (                  )', fontsize=lbl_sz)
    
    
    if atmos_subset == "training": 
        ax[1].text(-147, 58, 'b)', va='center', ha='center', 
                    rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz, backgroundcolor='white', zorder=9)
    elif atmos_subset == "testing": 
        ax[1].text(-147, 58, 'e)', va='center', ha='center', 
                    rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz, backgroundcolor='white', zorder=9)
    
    
    ##### Plot 3: W-Winds, Specific Humidity
    ax[2].contour(loni, lati, wwnd700ANOM, colors='blue', 
                  levels=wwnd_levs, transform=datacrs, zorder=5)
    # shum_ax = ax[2].contourf(loni, lati, shum850, cmap=shum_cmap, extend='max',
    #               levels=shum_levs, transform=datacrs, zorder=1)
    shum_ax = ax[2].contourf(loni, lati, shum850ANOM, cmap=shum_cmap, extend='both',
                             levels=shum_levs, transform=datacrs, zorder=1)
    
    # Colorbar
    cb_shum = plt.colorbar(shum_ax, pad=pdng, aspect=apct, fraction=frc, 
                            orientation='horizontal', ticks=[-1, 0, 1])
    cb_shum.set_label("850 hPa q Anom.", size=lbl_sz)
    cb_shum.ax.tick_params(labelsize=lbl_sz)
    
    # Cartography Features
    ax[2].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=2, edgecolor=border_color)
    ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), zorder=3, edgecolor=border_color)
    ax[2].add_feature(states_provinces, zorder=4, edgecolor=border_color)
    ax[2].set_extent(extent)
    
    # if clust+1 == 2:
    #     hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
    #                             transform=datacrs, hatches=[None, None, None, '..', 'oo'], 
    #                             levels=[0, 0.2, 0.4, 0.6, 0.8, 1.00])
    if clust+1 == 1:
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, '..', '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 2:
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, '..', '..', '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 3:
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 4:
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    elif clust+1 == 5:
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    else:
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, None, None, None, None, None, None, None, '..', 'oo'], 
                                levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.00])
    
    ax[2].set_title('700 hPa W-Wind Anom.' + '\n' + 
                    '(contour, +0.5 $\sigma$)', fontsize=lbl_sz)

    if atmos_subset == "training": 
        ax[2].text(-147, 58, 'c)', va='center', ha='center', 
                    rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz, backgroundcolor='white', zorder=9)
    elif atmos_subset == "testing": 
        ax[2].text(-147, 58, 'f)', va='center', ha='center', 
                    rotation_mode='anchor', transform=datacrs, 
                    fontsize=lbl_sz, backgroundcolor='white', zorder=9)

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    # # Add Title with Cluster Number
    # n_events = str(np.shape(atmos_time)[0])
    # plt.suptitle(f"Cluster {str(clust+1)}: {n_events} Days", fontsize=24, y=1.01, weight='bold')   

    # Add Colorbar for Hatching
    cbar_cam = fig.add_axes([0.91, 0.2, 0.02, 0.5]) #[x0, y0, width, height]
    cb_cam = fig.colorbar(hatch, cax=cbar_cam, pad=pdng, 
                          fraction=0.032, orientation='vertical')
    cb_cam.set_label("Max. Grad-CAM", size=lbl_sz)
    cb_cam.ax.tick_params(labelsize=lbl_sz)
    
    # create path to save
    path = model_dir + model_name + "/kmeans/"
    if os.path.exists(path) == False:
        os.mkdir(path)

    # plt.savefig(f"{path}kmeans_cluster_{str(clust+1)}_{atmos_subset}_atmos_fixed_hatchGradient_wnds.png", dpi=200, 
    #             transparent=True, bbox_inches='tight')
    
    plt.show()
    
    plt.clf()



#%% plot presets

IVT_cmap = ncm.cmap('WhiteYellowOrangeRed')
IVT_levs = np.arange(50, 500, 50)

ANOM_levs = np.arange(-5, 5.5, 0.5)

uwnd_scl = 300
uwnd_col = [0.1, 0.1, 0.1]

pr_cmap = ncm.cmapDiscrete('prcp_2', indexList=np.arange(1,12))
pr_levs = np.arange(0.5, 6.5, 0.5)

temp_cmap = ncm.cmapDiscrete('BlRe', indexList=[24,24,24,24,24,24,24, 72,72,72,72,72,72])
temp_levs = np.arange(-32, 40, 4)

vwnd_scl = 125

wwnd_cmap = ncm.cmap('BlueDarkRed18')
wwnd_levs = np.array([1, 10])

shum_cmap = ncm.cmap('MPL_BrBG')
shum_levs = np.arange(-3, 3.5, 0.5)

border_color = [0.45, 0.45, 0.45]

apct = 14
pdng = 0.03
lbl_sz = 14
frc = 0.15

gradcam_cmap = ncm.cmap('MPL_Greys')




#%% Loop through each day and plot daily Grad-CAM

cluster_to_plot = [0,1,3]

def grad_cam_daily(atmos_clust):
    
    # needed vars
    opg_sz = dict(opg.sizes)
    
    
    # Calculate Size of Data
    atm_sz = dict(atmos_clust.sizes)
    opg_sz = dict(opg.sizes)
    atmos_names = np.array(atmos_clust.data_vars)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos_clust.values():
        if len(ii.dims) == 2:
            count_channels += 1
        elif len(ii.dims) == 1:
            count_on_facet += 1
            atmos_names = np.delete(
                atmos_names, np.argmax(ii.name == atmos_names))
    
    
    # Create Empty Array for Testing Data
    atmos_test_4D_X = np.zeros((atm_sz['lat'], atm_sz['lon'], count_channels))
    atmos_test_OF_X = np.zeros((1, opg_sz['facet_num']*count_on_facet))

    # Pull Variables for Testing
    count_4D = 0
    count_OF = 0
    for var_name, values in atmos_clust.items():
        if len(values.dims) == 2:
            atmos_test_4D_X[ :, :, count_4D] = np.array(
                values[:, :])
            count_4D += 1
        elif len(values.dims) == 1:
            atmos_test_OF_X[:, opg_sz['facet_num']*count_OF:opg_sz['facet_num']
                            * (count_OF+1)] = np.array(values[:, :])
            count_OF += 1

    # Combine Inputs if On-Facet Data Exists
    if count_on_facet > 0:
        inputs = [atmos_test_4D_X, atmos_test_OF_X]
    else:
        inputs = [atmos_test_4D_X[tf.newaxis,...]]
    
    # Calculate the conposite grad-cam
    heatmap = []

    for output_node in range(opg_sz['facet_num']):
        # Create a model that takes the input image and outputs the last
        # convolutional layer and the output predictions
        grad_model = tf.keras.models.Model([model.inputs],
                                           [model.get_layer('conv2d_3').output, model.output])

        # Then compute the gradient of the top predicted class for our
        # input image with resepct to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(inputs)
            class_channel = preds[:, output_node]

        # This is the gradient of the output neuron with regard to the
        # output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Formulates the mean gradient of each feature map, how important is each feature map?
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Reduce to 3 dimentions
        last_conv_layer_output = last_conv_layer_output[0]

        # Multiply each channel of the feature map by how important it is
        heatmapx = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

        # Reduce dimentions
        # heatmapx = np.mean(heatmapx, axis=2)
        heatmap.append(np.array(heatmapx))

    heatmap = np.concatenate(heatmap, axis=2)  # combine all maps
    heatmap = np.max(heatmap, axis=2) # create max of all maps
    # create percentile of all maps
    # heatmap = np.percentile(heatmap, 90, axis=2)

    heatmap = heatmap / np.max(heatmap[:])  # Scale to max of 1
    heatmap[heatmap < 0] = 0  # Remove all the negative values
    
    return heatmap



for clust in cluster_to_plot:
    
    _, atmos_clust, atmos_time = atmos_cluster(clust)
    
    for dayx in range(len(atmos_clust.time.values)):
        
        atmos_clust_day = atmos_clust.isel(time=dayx)
    
        grad_cam_clust = grad_cam_daily(atmos_clust_day)
    
        # pull plotting axis values
        loni, lati = np.meshgrid(atmos['lon'], atmos['lat'])
        
        # find size of figure
        atmos_name = [i for i in atmos_clust_day.data_vars]
        
        # interpolate the heatmap grid
        grad_cam_clust_size = np.shape(grad_cam_clust)
        heatmap_lat = np.linspace(
            np.min(atmos['lat']), np.max(atmos['lat']), num=grad_cam_clust_size[0])
        heatmap_lon = np.linspace(
            np.min(atmos['lon']), np.max(atmos['lon']), num=grad_cam_clust_size[1])
        h_lon, h_lat = np.meshgrid(heatmap_lon, heatmap_lat)
        
        # Plot Transforms
        datacrs = ccrs.PlateCarree()
        projex = ccrs.Mercator(central_longitude=np.mean(atmos['lon'].values))
        
        # Change vars
        # hgt500 = atmos_clust_mean.hgt500.values / 9.81
        
        hgt500ANOM = np.array(atmos_clust_day.hgt500)
        temp700ANOM = np.array(atmos_clust_day.temp700)
        shum850ANOM = np.array(atmos_clust_day.shum850)
        wwnd700ANOM = np.array(atmos_clust_day.wwnd700)
        
        atmos_clust_act = (atmos_clust_day * atmos_std) + atmos_mean
        IVTsfc = atmos_clust_act.IVTsfc.values
        IVTsfc[IVTsfc < 100] = np.nan
        # temp700 = atmos_clust_mean.temp700.values - 273.15
        precipsfc = atmos_clust_act.precipsfc.values * 1000
        precipsfc[precipsfc < 0.5] = np.nan
        shum850 = atmos_clust_act.shum850.values * 1000
        shum850[shum850 < 0.1] = np.nan
        
        # new figure like michaels
        fig, ax = plt.subplots(nrows=1, ncols=x_axis,
                               figsize=(x_axis*4, 7),
                               subplot_kw={'projection': projex})
        
        ##### Plot 1: Geo Height, IVT, U-Winds
        # geo_lbl = ax[0].contour(loni, lati, hgt500, colors='black', 
        #               transform=datacrs, zorder=6)
        # ax[0].clabel(geo_lbl, fmt="%.0f", inline=True)
        
        geo_lbl = ax[0].contour(loni, lati, hgt500ANOM, levels=ANOM_levs,
                      cmap=temp_cmap, transform=datacrs, zorder=6, 
                      linestyles=np.where(ANOM_levs > 0, "-", "--"))
        ax[0].clabel(geo_lbl, fmt="%.2f", inline=True)
        
        
        IVT_ax = ax[0].contourf(loni, lati, IVTsfc, cmap=IVT_cmap, 
                       levels=IVT_levs, transform=datacrs, extend='max', zorder=2)
        ax[0].quiver(loni[::7, ::10], lati[::7, ::10], atmos_clust_act.uwnd700.values[::7, ::10], 
                     np.zeros(np.shape(atmos_clust_act.uwnd700.values))[::7, ::10], 
                     scale=uwnd_scl, pivot='mid', transform=datacrs, zorder=7,
                     color=uwnd_col, headwidth=6)
        
        # Colorbar
        cb_IVT = plt.colorbar(IVT_ax, extend='max', pad=pdng, aspect=apct, fraction=frc, 
                              orientation='horizontal', ticks=[100,200,300,400])
        cb_IVT.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", size=lbl_sz)
        cb_IVT.ax.tick_params(labelsize=lbl_sz)
        
        # Cartography Features
        states_provinces = cfeat.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')
        ax[0].add_feature(cfeat.LAND, facecolor='lightgray', zorder=1)
        ax[0].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=3, edgecolor=border_color)
        ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), zorder=4, edgecolor=border_color)
        ax[0].add_feature(states_provinces, zorder=5, edgecolor=border_color)
        ax[0].set_extent(extent)
        
        # hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
        #                         transform=datacrs, hatches=[None, '..', 'oo','OO'], 
        #                         levels=[0, 0.25, 0.50, 0.75, 1.00])
        
        hatch = ax[0].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, '..', 'oo'], 
                                levels=[0, 0.3, 0.6, 1.00])
        
        
        ax[0].set_title('500 hPa Hgt. Anom. (contour)' + '\n' + 
                        '700 hPa U-Winds (vector, m s$^{-1}$)', fontsize=lbl_sz)
        
        if atmos_subset == "training": 
            ax[0].text(-151, 42.5, 'Training', va='bottom', ha='center', 
                       rotation='vertical', rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz)
            ax[0].text(-147, 58, 'a)', va='center', ha='center', 
                       rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz, backgroundcolor='white', zorder=9)
        elif atmos_subset == "testing": 
            ax[0].text(-151, 42.5, 'Testing', va='bottom', ha='center', 
                       rotation='vertical', rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz)
            ax[0].text(-147, 58, 'd)', va='center', ha='center', 
                       rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz, backgroundcolor='white', zorder=9)
        
        ##### Plot 2: Precip, temp, v-winds
        # temp_lbl = ax[1].contour(loni, lati, temp700, levels=temp_levs,
        #               cmap=temp_cmap, transform=datacrs, zorder=6, 
        #               linestyles=np.where(temp_levs > 0, "-", "--"))
        # ax[1].clabel(temp_lbl, fmt="%.0f", inline=True, fontsize=lbl_sz)
        
        temp_lbl = ax[1].contour(loni, lati, temp700ANOM, levels=ANOM_levs,
                      cmap=temp_cmap, transform=datacrs, zorder=6, 
                      linestyles=np.where(ANOM_levs > 0, "-", "--"))
        ax[1].clabel(temp_lbl, fmt="%.2f", inline=True)  
        
        
        pr_ax = ax[1].contourf(loni, lati, precipsfc, cmap=pr_cmap, 
                       levels=pr_levs, transform=datacrs, extend='max', zorder=2)
        ax[1].quiver(loni[::10, ::7], lati[::10, ::7], np.zeros(np.shape(atmos_clust_act.vwnd_10msfc.values))[::10, ::7],
                     atmos_clust_act.vwnd_10msfc.values[::10, ::7], scale=vwnd_scl, 
                     transform=datacrs, zorder=7, headwidth=6)
        
        # Colorbar
        cb_pr = plt.colorbar(pr_ax, pad=pdng, aspect=apct, fraction=frc, 
                             orientation='horizontal', extend='max', 
                             ticks=[1,2,3,4,5,6])
        cb_pr.set_label("Precipitation (mm)", size=lbl_sz)
        cb_pr.ax.tick_params(labelsize=lbl_sz)
        
        # Cartography Features
        ax[1].add_feature(cfeat.LAND, facecolor='lightgray', zorder=1)
        ax[1].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=3, edgecolor=border_color)
        ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), zorder=4, edgecolor=border_color)
        ax[1].add_feature(states_provinces, zorder=5, edgecolor=border_color)
        ax[1].set_extent(extent)
        
        hatch = ax[1].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=8, 
                                transform=datacrs, hatches=[None, '..', 'oo'], 
                                levels=[0, 0.3, 0.6, 1.00])
        
        ax[1].set_title('700 hPa Temp. Anom. (contour)' + '\n' + 
                        '10-m V-Winds (vector, m s$^{-1}$)', fontsize=lbl_sz)
        
        
        if atmos_subset == "training": 
            ax[1].text(-147, 58, 'b)', va='center', ha='center', 
                       rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz, backgroundcolor='white', zorder=9)
        elif atmos_subset == "testing": 
            ax[1].text(-147, 58, 'e)', va='center', ha='center', 
                       rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz, backgroundcolor='white', zorder=9)
        
        
        ##### Plot 3: W-Winds, Specific Humidity
        ax[2].contour(loni, lati, wwnd700ANOM, colors='blue', 
                      levels=wwnd_levs, transform=datacrs, zorder=5)
        shum_ax = ax[2].contourf(loni, lati, shum850ANOM, cmap=shum_cmap, extend='both',
                      levels=shum_levs, transform=datacrs, zorder=1)
        
        # Colorbar
        cb_shum = plt.colorbar(shum_ax, pad=pdng, aspect=apct, fraction=frc, 
                               orientation='horizontal', ticks=[-3, -2, -1, 0, 1, 2, 3])
        cb_shum.set_label("850 hPa q Anom.", size=lbl_sz)
        cb_shum.ax.tick_params(labelsize=lbl_sz)
        
        # Cartography Features
        ax[2].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=2, edgecolor=border_color)
        ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), zorder=3, edgecolor=border_color)
        ax[2].add_feature(states_provinces, zorder=4, edgecolor=border_color)
        ax[2].set_extent(extent)
        
        hatch = ax[2].contourf(h_lon, h_lat, grad_cam_clust, colors='none', zorder=6, 
                                transform=datacrs, hatches=[None, '..', 'oo'], 
                                levels=[0, 0.3, 0.6, 1.00])
        
        ax[2].set_title('700 hPa W-Wind' + '\n' + 
                        '(contour, +1 $\sigma$)', fontsize=lbl_sz)
    
        if atmos_subset == "training": 
            ax[2].text(-147, 58, 'c)', va='center', ha='center', 
                       rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz, backgroundcolor='white', zorder=7)
        elif atmos_subset == "testing": 
            ax[2].text(-147, 58, 'f)', va='center', ha='center', 
                       rotation_mode='anchor', transform=datacrs, 
                       fontsize=lbl_sz, backgroundcolor='white', zorder=7)
    
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        
        # # Add Title with Cluster Number
        # n_events = str(np.shape(atmos_time)[0])
        # plt.suptitle(f"Cluster {str(clust+1)}: {n_events} Days", fontsize=24, y=1.01, weight='bold')   
    
        # Add Colorbar for Hatching
        cbar_cam = fig.add_axes([0.91, 0.2, 0.02, 0.5]) #[x0, y0, width, height]
        cb_cam = fig.colorbar(hatch, cax=cbar_cam, pad=pdng, 
                              fraction=0.032, orientation='vertical')
        cb_cam.set_label("Max. Grad-CAM", size=lbl_sz)
        cb_cam.ax.tick_params(labelsize=lbl_sz)
        
        ttx = pd.to_datetime(atmos_clust.time.values[dayx]).strftime('%Y-%m-%d')
        
        # create path to save
        path = model_dir + model_name + "/kmeans/daily/"
        if os.path.exists(path) == False:
            os.mkdir(path)
    
        # plt.savefig(f"{path}kmeans_cluster_{str(clust+1)}_{atmos_subset}_{ttx}.png", dpi=200, 
        #             transparent=True, bbox_inches='tight')
        
        # plt.show()
        
        plt.close()

