"""
Savanna Wolvin
Created: Jul 28th, 2023
Edited: Jul 28th, 2023

##### Summary ################################################################



##### Input ##################################################################



##### Output #################################################################




"""
#%% Global Imports

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import timedelta, datetime
import scipy.io as sio
import scipy.stats as sp_stats
from matplotlib.ticker import MultipleLocator
import os
import sys
sys.path.insert(1, '/uufs/chpc.utah.edu/common/home/u1324060/nclcmappy/')
import nclcmaps as ncm




#%% Preset Variables

model_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_facet_cnn_weighting/NR/"
model_name = "2024-01-08_1113"

kmeans_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_kmeans_clust/"

opg_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/opg/"

fi_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/facets/"

units = "mm/m"
min_val = -0.06
max_val = 0.11
max_heat = 1000

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
output_test = xr.open_dataset(f"{model_dir}{model_name}/stats/Testing_output_stats.nc", engine='netcdf4')
output_train = xr.open_dataset(f"{model_dir}{model_name}/stats/Training_output_stats.nc", engine='netcdf4')
# actual = output.actual.values
# predicted = output.predicted.values


# Load Kmeans Data
kmeans = pd.read_csv(f"{kmeans_dir}kmeans_clusters_opg_NR")

# Load actual OPG
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




#%% Formulate all variables

# Training Predicted, Testing Predicted, Actual

opg_mean    = np.zeros((np.max(kmeans['cluster'].values)+1,3))
opg_std     = np.zeros((np.max(kmeans['cluster'].values)+1,3))
r2          = np.zeros((np.max(kmeans['cluster'].values)+1,2))
mae         = np.zeros((np.max(kmeans['cluster'].values)+1,2))
mre         = np.zeros((np.max(kmeans['cluster'].values)+1,2))
me          = np.zeros((np.max(kmeans['cluster'].values)+1,2))

for clust in np.unique(kmeans['cluster'].values):
    
    # Pull Dates with that cluster
    clust_dates = kmeans['datetime'][kmeans['cluster']==clust]
    
    ## Pull training values ###################################################
    
    # Pull Dates within the Cluster
    output_time = pd.to_datetime(output_train.time.values)
    output_time = output_time.drop(output_time.difference(clust_dates))
    
    # Pull selected Atmos Dates and Take Time Mean
    output_clust = output_train.sel(time=output_time)
    
    # Pull Values for the Cluster
    actual    = np.reshape(output_clust.actual.values, -1)
    predicted = np.reshape(output_clust.predicted.values, -1)
    # idx       = [actual > 0.00001] or [actual < -0.00001]
    idx       = actual != 0.0
    actual    = actual[idx]
    predicted = predicted[idx]
    
    r2[clust, 0] = np.around(np.corrcoef(actual, predicted)**2, decimals=4)[0,1]
    mae[clust, 0] = np.nanmean(np.abs(actual - predicted))
    me[clust, 0] = np.around(np.nanmean(actual - predicted), decimals=5)
    mre[clust, 0] = np.around(np.nanmean(np.abs(actual - predicted)/actual), decimals=5)
    opg_mean[clust, 0] = np.mean(predicted)
    opg_std[clust, 0] = np.std(predicted)
    
    ## Pull testing values ####################################################
    
    # Pull Dates within the Cluster
    output_time = pd.to_datetime(output_test.time.values)
    output_time = output_time.drop(output_time.difference(clust_dates))
    
    # Pull selected Atmos Dates and Take Time Mean
    output_clust = output_test.sel(time=output_time)
    
    # Pull Values for the Cluster
    actual    = np.reshape(output_clust.actual.values, -1)
    predicted = np.reshape(output_clust.predicted.values, -1)
    # idx       = [actual > 0.00001] or [actual < -0.00001]
    idx       = actual != 0.0
    actual    = actual[idx]
    predicted = predicted[idx]
    
    r2[clust, 1] = np.around(np.corrcoef(actual, predicted)**2, decimals=4)[0,1]
    mae[clust, 1] = np.nanmean(np.abs(actual - predicted))
    me[clust, 1]  = np.around(np.nanmean(actual - predicted), decimals=5)
    mre[clust, 1] = np.around(np.nanmean(np.abs(actual - predicted)/actual), decimals=5)
    opg_mean[clust, 1] = np.mean(predicted)
    opg_std[clust, 1] = np.std(predicted)
    
    ## Pull All Actual Values
    opgs = facet_opg.opg.values
    opgs[opgs==0] = np.nan
    opg_mean[clust, 2] = np.nanmean(opgs[kmeans['cluster']==clust,:])
    opg_std[clust, 2] = np.nanstd(opgs[kmeans['cluster']==clust,:])





#%% Bar plot

lbl_sz = 16
tt_sz = 18

ccmap = plt.get_cmap('nipy_spectral', 11)

fig, ax = plt.subplots(layout='constrained', nrows=2, ncols=1, sharex=True, 
                       height_ratios=[0.75, 0.25], figsize=(6, 7))
x_range = np.arange(1,6)

# # Scatter plot of MRE
# ax[2].scatter(x_range, mre[:,0], s=150, edgecolors='black', 
#             fc='royalblue', label='Training')
# ax[2].scatter(x_range, mre[:,1], s=150, edgecolors='black', 
#             fc='orangered', label='Testing')
# ax[2].set_ylim(0,3)
# #ax[2].set_ylabel('MRE with Actual')
# ax[2].set_title('c) MRE with Actual')
# # ax[1].legend(loc='upper right', ncols=2)
# ax[2].grid(True, which='major', color='dimgrey')
# ax[2].set_axisbelow(True)
# ax[2].yaxis.set_minor_locator(MultipleLocator(0.5))
# ax[2].grid(True, which='both')



# Scatter plot of r2
ax[1].scatter(x_range, r2[:,0], s=150, edgecolors='black', 
            fc='royalblue', label='Training')
ax[1].scatter(x_range, r2[:,1], s=150, edgecolors='black', 
            fc='orangered', label='Testing')
ax[1].set_ylim(0,1)
#ax[1].set_ylabel('$\mathregular{r^2}$ with Actual')
ax[1].set_title('b) $\mathregular{r^2}$ Between Actual and Predicted', size=tt_sz)
#ax[1].legend(loc='upper right', ncols=2)
ax[1].grid(True, which='major', color='dimgrey')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))
ax[1].grid(True, which='both')


# Bar plots of mean OPG and MAE
ax[0].bar(x_range-0.25, opg_mean[:,2], 0.25, label='Actual', fc='black')
ax[0].bar(x_range, opg_mean[:,0], 0.25, label='Training', fc='royalblue')
ax[0].bar(x_range+0.25, opg_mean[:,1], 0.25, label='Testing', fc='orangered')

ax[0].scatter(x_range, mae[:,0], s=150, edgecolors='black', 
           facecolor='None', label='MAE to Actual OPG')
ax[0].scatter(x_range+0.25, mae[:,1], s=150, edgecolors='black', 
           facecolor='None')

#ax[0].set_ylabel('OPG (mm/km)')
ax[0].set_title('a) Mean OPG (mm/m)', size=tt_sz)
ax[0].set_xticks(x_range, fonstize=lbl_sz)
ax[1].set_xlabel('Regional Daily Winter OPG Event Clusters', size=lbl_sz)
ax[0].legend(loc='upper right', ncols=1, fontsize=lbl_sz)
ax[0].set_ylim(0,0.0105)
ax[0].set_xlim(np.min(x_range)-0.5,np.max(x_range)+0.5)

ax[0].grid(True, which='major', color='dimgrey')
ax[0].set_axisbelow(True)
ax[0].yaxis.set_minor_locator(MultipleLocator(0.001))
ax[0].grid(True, which='both')


ax[0].tick_params(labelsize=lbl_sz)
ax[1].tick_params(labelsize=lbl_sz)

# create path to save
path = model_dir + model_name + "/kmeans/"
if os.path.exists(path) == False:
    os.mkdir(path)

#plt.show()

plt.savefig(f"{path}kmeans_cluster_stats_bar_scatter.png", dpi=300, 
            transparent=True, bbox_inches='tight')







#%% Try a plot out

ccmap = plt.get_cmap('nipy_spectral', 11)
cccmap = [[0,	0,	0],
        [0.5333,	0,	0.6],
        [0,	0,	0.8667],
        [0,	0.6,	0.8667],
        [0,	0.6667,	0.5333],
        [0,	0.7333,	0],
        [1.3026e-15,	1,	0],
        [0.9333,	0.9333,	0],
        [1,	0.6,	0],
        [0.8667,	0,	0],
        [0.8,	0.8,	0.8]]

plt.plot(r2[0,:], mae[0,:]*1000, c=cccmap[0], linestyle=':', zorder=1)
plt.plot(r2[1,:], mae[1,:]*1000, c=cccmap[1], linestyle=':', zorder=2)
plt.plot(r2[2,:], mae[2,:]*1000, c=cccmap[2], linestyle=':', zorder=3)
plt.plot(r2[3,:], mae[3,:]*1000, c=cccmap[3], linestyle=':', zorder=4)
plt.plot(r2[4,:], mae[4,:]*1000, c=cccmap[4], linestyle=':', zorder=5)
plt.plot(r2[5,:], mae[5,:]*1000, c=cccmap[5], linestyle=':', zorder=6)
plt.plot(r2[6,:], mae[6,:]*1000, c=cccmap[6], linestyle=':', zorder=7)
plt.plot(r2[7,:], mae[7,:]*1000, c=cccmap[7], linestyle=':', zorder=8)
plt.plot(r2[8,:], mae[8,:]*1000, c=cccmap[8], linestyle=':', zorder=9)
plt.plot(r2[9,:], mae[9,:]*1000, c=cccmap[9], linestyle=':', zorder=10)
plt.plot(r2[10,:], mae[10,:]*1000, c=cccmap[10], linestyle=':', zorder=11)

plt.scatter(r2[:,0], mae[:,0]*1000, s=100, c=range(1,12), cmap=ccmap, edgecolors='black', zorder=12)
plt.scatter(r2[:,1], mae[:,1]*1000, s=100, c=range(1,12), marker='<', cmap=ccmap, edgecolors='black', zorder=13)

plt.xlim([0,0.8])
plt.ylim([0, 6])

plt.xlabel('$\mathregular{r^2}$')
plt.ylabel('MAE (mm/km)')

plt.grid()

plt.colorbar(ticks=range(1,12), label='Cluster Number', pad=0.01)
plt.clim(0.5,11.5)

plt.show()





#%% Create stem plot

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(3, 10))

ax[0].scatter(range(1,12), opg_mean[:,0]*1000, c='blue')
ax[0].scatter(range(1,12), opg_mean[:,1]*1000, c='darkorange')
ax[0].scatter(range(1,12), opg_mean[:,2]*1000, c='black')
ax[0].set_ylim([0,12.5])
ax[0].grid()
ax[0].set_title('OPG Mean')

ax[1].scatter(range(1,12), opg_std[:,0]*1000, c='blue')
ax[1].scatter(range(1,12), opg_std[:,1]*1000, c='darkorange')
ax[1].scatter(range(1,12), opg_std[:,2]*1000, c='black')
ax[1].set_ylim([0,12.5])
ax[1].grid()
ax[1].set_title('OPG Standard Deviation')


ax[2].scatter(range(1,12), r2[:,0], c='blue')
ax[2].scatter(range(1,12), r2[:,1], c='darkorange')
ax[2].set_ylim([0,1])
ax[2].grid()
ax[2].set_title('r^2')

ax[3].scatter(range(1,12), mae[:,1]*1000, c='darkorange')
ax[3].scatter(range(1,12), mae[:,0]*1000, c='blue')
ax[3].set_ylim([0,6])
ax[3].grid()
ax[3].set_title('MAE')

plt.show()


#%%


    
    # # create path to save
    # path = model_dir + model_name + "/kmeans/"
    # if os.path.exists(path) == False:
    #     os.mkdir(path)

    # plt.savefig(f"{path}kmeans_cluster_{str(clust+1)}_{atmos_subset}_actVSpred.png", dpi=200, 
    #             transparent=True, bbox_inches='tight')
    
    # plt.show()










































