"""
Savanna Wolvin
Created: Jul 28th, 2023
Edited: Jul 28th, 2023

##### Summary ################################################################



##### Input ##################################################################



##### Output #################################################################




"""
#%% Global Imports

import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeat
import scipy.io as sio
import scipy.stats as sp_stats
from datetime import timedelta, datetime
import os
import sys
sys.path.insert(1, '/uufs/chpc.utah.edu/common/home/u1324060/nclcmappy/')
import nclcmaps as ncm



#%% Preset Variables

model_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_facet_cnn_FINAL/NR/"
model_name = "2023-06-05_1443"

atmos_subset = "training"

kmeans_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_kmeans_clust/"

fi_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/facets/"

opg_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/opg/"

years           = [1979, 2018]
months          = [12, 1, 2]
num_station = 3

# CR - Coastal Ranges; NR - Northern Rockies; IPN - Inland Pacific Northwest; 
# SW - Southwest; SR - Southern Rockies
fi_region    = "NR"
# How to Handle NaN's? 0 - remove all NaN days, 1 - Set NaNs to zero, 2 - Set NaNs to average
opg_nans        = 1
# Type of OPG Values to Use? 0 - raw OPG, 1 - standardized, 2 - normalized
opg_type        = 1
# Percent of Days Within Timeseries the Facet Must Have Observations
prct_obs        = 0.2
# Minimum number of stations on the Facet to use the OPG value
num_station = 3


#%% Load Data

# Load Output Stats
train_output = xr.open_dataset(f"{model_dir}{model_name}/stats/Training_output_stats.nc", engine='netcdf4')
test_output = xr.open_dataset(f"{model_dir}{model_name}/stats/Testing_output_stats.nc", engine='netcdf4')

# Load Kmeans Data
kmeans = pd.read_csv(f"{kmeans_dir}kmeans_clusters_opg_NR")

# Load OPG Data
mat_file = sio.loadmat(fi_dir + 'lats')
lats  = mat_file['lats']

mat_file = sio.loadmat(fi_dir + 'lons')
lons  = mat_file['lons']

mat_file = sio.loadmat(fi_dir + 'facets')
orientation  = mat_file['facets']

mat_file = sio.loadmat(fi_dir + 'facets_labeled')
facets  = mat_file['facets_i']

time_ref    = np.arange(datetime(1979, 1, 1), datetime(2018, 4, 1), 
                            timedelta(days=1), dtype='object')
fi_num    = np.array(range(1,np.max(facets)+1))
mat_file = sio.loadmat(opg_dir + 'all_opg')  
facet_opg   = xr.Dataset()
facet_opg['opg'] = xr.DataArray(data=mat_file['allOPG_qc2'], 
                                coords=[time_ref, fi_num], 
                                dims=['time','facet_num'])



#%% Plot Presets 

x_axis = 3
cmap_feat = ncm.cmap('NCV_blu_red')
extent = [-150, -100, 25, 57.5]
levs = np.arange(-2, 2.1, 0.1)

# load regional location
bounds = [40.5, 49.5, -116, -105]



#%% Pull Desired Domain

# Create Array of All Desired Years
d_years = np.arange(years[0], years[1]+1).astype('int')


# calculate the mean lat/lons to determine domain region
fi_idx = np.zeros([np.max(facets)], dtype=bool)
for fi in range(1,np.max(facets)+1):
    mlat, mlon = np.mean([lats[facets == fi],lons[facets == fi]], axis=1)
    orient = sp_stats.mode(orientation[facets == fi], keepdims=False)
    
    if orient[0] != 9: # check if its flat
        if mlat>36 and mlat<49 and mlon>-125 and mlon<-122 and fi_region=="CR":
            fi_idx[fi-1] = True
            bounds = [36, 49, -125, -122]
        elif mlat>31 and mlat<40.5 and mlon>-111 and mlon<-103 and fi_region=="SR":
            fi_idx[fi-1] = True
            bounds = [31, 40.5, -111, -103]
        elif mlat>31 and mlat<40.5 and mlon>-122 and mlon<-111 and fi_region=="SW":
            fi_idx[fi-1] = True
            bounds = [31, 40.5, -122, -111]
        elif mlat>40.5 and mlat<49.5 and mlon>-122 and mlon<-116 and fi_region=="IPN":
            fi_idx[fi-1] = True
            bounds = [40.5, 49.5, -122, -116]
        elif mlat>40.5 and mlat<49.5 and mlon>-116 and mlon<-105 and fi_region=="NR":
            fi_idx[fi-1] = True
            bounds = [40.5, 49.5, -116, -105]

fi_numX    = np.array(range(1,np.max(facets)+1))
fi_num     = fi_numX[fi_idx]
non_fi_num = fi_numX[~fi_idx]

# Set OPG to zero if there are not enough stations
mat_file = sio.loadmat(fi_dir + 'station_count')  
rm_opg   = mat_file['num'] < num_station
facet_opg.opg.values[rm_opg] = 0

# Drop Facets Outside of Region
facet_opg   = facet_opg.drop_sel(facet_num=(non_fi_num))
facet_opg.attrs["bounds"] = bounds

# Pull Data from Specified Years and Months
facet_opg = facet_opg.isel(time=(facet_opg.time.dt.year.isin(d_years)))
facet_opg = facet_opg.isel(time=(facet_opg.time.dt.month.isin(months)))

# Drop Facets Without Observations
opg_nan_idx = facet_opg['opg'].sum(dim = 'time', skipna = True) == 0
facet_opg   = facet_opg.drop_sel(facet_num=(fi_num[opg_nan_idx]))

# Drop Facets With Not Enough Observations
fi_num      = facet_opg.facet_num.values
if_opg      = (np.sum(facet_opg.opg.values > 0, axis=0) + np.sum(facet_opg.opg.values < 0, axis=0)) / np.shape(facet_opg.time.values)[0]
facet_opg   = facet_opg.drop_sel(facet_num=(fi_num[if_opg < prct_obs]))

# Decide How To Handle NaNs
if opg_nans == 0: ##### Remove Dates With NaN Values #####################
    real_opg_dates = facet_opg.time[~pd.isna(facet_opg.opg.values)]
    facet_opg = facet_opg.where(facet_opg.time==real_opg_dates)
elif opg_nans == 1: ##### Set All NaNs to Zero ###########################
    facet_opg.opg.values[pd.isna(facet_opg.opg.values)] = 0
    # facet_opg.opg.values[pd.isna(facet_opg.opg.values)] = -999
elif opg_nans == 2: ##### Set All NaNs to the Average Value ##############
    facet_opg.opg.values[pd.isna(facet_opg.opg.values)] = facet_opg.opg.mean()




#%% Formulate all the cluster means

opg_matrix = np.zeros((np.max(kmeans['cluster'].values)+1, np.shape(facets)[0], np.shape(facets)[1]))

for clust in np.unique(kmeans['cluster'].values):
    
    # Take Mean of OPGs by Cluster
    opgs = facet_opg.opg.values
    opgs[opgs==0] = np.nan
    clust_opg = np.nanmean(opgs[kmeans['cluster']==clust,:], axis=0)
    #clust_opg = np.nanstd(opgs[kmeans['cluster']==clust,:], axis=0)
    
    # Pull Dates with that cluster
    clust_dates = kmeans['datetime'][kmeans['cluster']==clust]
    
    # Create Matrix
    opg_matrixx = np.zeros(np.shape(facets))
    opg_matrixx[:] = np.nan
    
    for fi in range(len(facet_opg.facet_num)):
        opg_matrixx[facets==facet_opg.facet_num[fi].values] = clust_opg[fi]
        
    opg_matrix[clust,:,:] = opg_matrixx




#%% Plot all clusters

y_axis = 2
x_axis = 3
datacrs = ccrs.PlateCarree()
projex = ccrs.Mercator(central_longitude=np.mean(lons))

# new figure like michaels
fig, ax = plt.subplots(nrows=y_axis, ncols=x_axis,
                        figsize=(x_axis*3, y_axis*4),
                        subplot_kw={'projection': projex})

varx = 0
for row in range(y_axis):
    for col in range(x_axis):
        if varx > (np.max(kmeans['cluster'].values)):
            ax[row, col].axis('off')
            
        else:
            
            ccmap = plt.get_cmap('bwr', 24)
            pcm1 = ax[row, col].pcolormesh(lons, lats, np.squeeze(opg_matrix[varx,:,:]), 
                                         cmap=ccmap, transform=datacrs, shading='auto')
            pcm1.set_clim(-0.015,0.015)
            
            # Add Regional Box
            ax[row, col].plot([bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]],
                      [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]],
                      transform=datacrs, color='black')
            
            # Cartography
            ax[row, col].add_feature(cfeat.LAND, facecolor="burlywood")
            ax[row, col].add_feature(cfeat.OCEAN)
            ax[row, col].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="saddlebrown")
            ax[row, col].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="saddlebrown")
            ax[row, col].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="saddlebrown")
            ax[row, col].set_extent([bounds[2]-1, bounds[3]+1, bounds[0]-1, bounds[1]+1])
            
            # Add Title
            ax[row, col].set_title("Cluster " + str(varx+1), size=20)
            
            varx += 1
            
# Fix Spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Add Colorbar for OPG
cbar_opg = fig.add_axes([0.14, 0.095, 0.74, 0.025]) #[x0, y0, width, height]
cb_opg = fig.colorbar(pcm1, cax=cbar_opg, ticks=[-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015],
                      extend='both', orientation='horizontal', pad=0.0, aspect=5, fraction=0.032)
cb_opg.set_ticklabels(['-0.015', '-0.01', '-0.005', '0.0', '0.005', '0.01', '0.015'])
cb_opg.set_label('Mean OPG (mm/m)', size=18)
cb_opg.ax.tick_params(labelsize=18)

# create path to save
path = model_dir + model_name + "/kmeans/"
if os.path.exists(path) == False:
    os.mkdir(path)
    
# Save and Show Figure
plt.savefig(f"{path}kmeans_clusters.png", dpi=300, 
            transparent=True, bbox_inches='tight')


plt.show()



#%% STD OF OPG

#%% Formulate all the cluster means

opg_matrix = np.zeros((np.max(kmeans['cluster'].values)+1, np.shape(facets)[0], np.shape(facets)[1]))

for clust in np.unique(kmeans['cluster'].values):
    
    # Take Mean of OPGs by Cluster
    opgs = facet_opg.opg.values
    opgs[opgs==0] = np.nan
    #clust_opg = np.nanmean(opgs[kmeans['cluster']==clust,:], axis=0)
    clust_opg = np.nanstd(opgs[kmeans['cluster']==clust,:], axis=0)
    
    
    # Pull Dates with that cluster
    clust_dates = kmeans['datetime'][kmeans['cluster']==clust]
    
    # Create Matrix
    opg_matrixx = np.zeros(np.shape(facets))
    opg_matrixx[:] = np.nan
    
    for fi in range(len(facet_opg.facet_num)):
        opg_matrixx[facets==facet_opg.facet_num[fi].values] = clust_opg[fi]
        
    opg_matrix[clust,:,:] = opg_matrixx




#% Plot all clusters

y_axis = 2
x_axis = 3
datacrs = ccrs.PlateCarree()
projex = ccrs.Mercator(central_longitude=np.mean(lons))

# new figure like michaels
fig, ax = plt.subplots(nrows=y_axis, ncols=x_axis,
                        figsize=(x_axis*3, y_axis*4),
                        subplot_kw={'projection': projex})

varx = 0
for row in range(y_axis):
    for col in range(x_axis):
        if varx > (np.max(kmeans['cluster'].values)):
            ax[row, col].axis('off')
            
        else:
            
            ccmap = plt.get_cmap('bwr', 24)
            pcm1 = ax[row, col].pcolormesh(lons, lats, np.squeeze(opg_matrix[varx,:,:]), 
                                         cmap=ccmap, transform=datacrs, shading='auto')
            pcm1.set_clim(0,0.016)
            
            # Add Regional Box
            ax[row, col].plot([bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]],
                      [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]],
                      transform=datacrs, color='black')
            
            # Cartography
            ax[row, col].add_feature(cfeat.LAND, facecolor="burlywood")
            ax[row, col].add_feature(cfeat.OCEAN)
            ax[row, col].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="saddlebrown")
            ax[row, col].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="saddlebrown")
            ax[row, col].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="saddlebrown")
            ax[row, col].set_extent([bounds[2]-1, bounds[3]+1, bounds[0]-1, bounds[1]+1])
            
            # Add Title
            ax[row, col].set_title("Cluster " + str(varx+1), size=18)
            
            varx += 1
            
# Fix Spacing
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Add Colorbar for OPG
cbar_opg = fig.add_axes([0.14, 0.095, 0.74, 0.025]) #[x0, y0, width, height]
cb_opg = fig.colorbar(pcm1, cax=cbar_opg, ticks=[0, 0.005, 0.01, 0.015],
                      extend='max', orientation='horizontal', pad=0.0, aspect=5, fraction=0.032)
cb_opg.set_ticklabels(['0.0', '0.005', '0.01', '0.015'])
cb_opg.set_label('Standard Deviation (mm/m)', size=16)
cb_opg.ax.tick_params(labelsize=16)

# create path to save
path = model_dir + model_name + "/kmeans/"
if os.path.exists(path) == False:
    os.mkdir(path)
    
# Save and Show Figure
plt.savefig(f"{path}kmeans_clusters_std.png", dpi=200, 
            transparent=True, bbox_inches='tight')


plt.show()