"""
Savanna Wolvin
Created: Jul 28th, 2023
Edited: May 24th, 2024

##### Summary ################################################################
This script is used to plot the mean OPGs, mean absolute error, and r2 of the 
clustered OPG events for a region of the western United States.


##### Input ##################################################################
model_dir       - Directory to model runs
model_name      - Directory to specific model run
kmeans_dir      - Directory to saved clusters
opg_dir         - Directory to the OPG value
fi_dir          - Directory to the Facet data
units           - OPG units
years           - Years to pull
months          - Months to pull
fi_region       - Region of focus
opg_nans        - How to Handle NaN OPGs
opg_type        - Type of OPGs used in training
prct_obs        - Percent of observations for a facet to be valid
num_station     - Minimum number of stations for an OPG observation to be valid



"""
#%% Global Imports

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import scipy.io as sio
import scipy.stats as sp_stats
from matplotlib.ticker import MultipleLocator
import os


#%% Preset Variables

model_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_facet_cnn_weighting/NR/"
model_name = "2024-01-08_1113"

kmeans_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_kmeans_clust/"

opg_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/opg/"

fi_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/facets/"

years           = [1979, 2018]
months          = [12, 1, 2]

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




#%% Plot presets

lbl_sz = 16
tt_sz = 18

act_color = 'black'
tra_color = 'royalblue'
tes_color = 'orangered'


#%% Bar plot

fig, ax = plt.subplots(layout='constrained', nrows=2, ncols=1, sharex=True, 
                       height_ratios=[0.75, 0.25], figsize=(6, 7))
x_range = np.arange(1,6)

# Scatter plot of r2
ax[1].scatter(x_range, r2[:,0], s=150, edgecolors='black', 
            fc=tra_color, label='Training')
ax[1].scatter(x_range, r2[:,1], s=150, edgecolors='black', 
            fc=tes_color, label='Testing')
ax[1].set_ylim(0,1)
#ax[1].set_ylabel('$\mathregular{r^2}$ with Actual')
ax[1].set_title('b) $\mathregular{r^2}$ Between Actual and Predicted', size=tt_sz)
#ax[1].legend(loc='upper right', ncols=2)
ax[1].grid(True, which='major', color='dimgrey')
ax[1].set_axisbelow(True)
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))
ax[1].grid(True, which='both')


# Bar plots of mean OPG and MAE
ax[0].bar(x_range-0.25, opg_mean[:,2], 0.25, label='Actual', fc=act_color)
ax[0].bar(x_range, opg_mean[:,0], 0.25, label='Training', fc=tra_color)
ax[0].bar(x_range+0.25, opg_mean[:,1], 0.25, label='Testing', fc=tes_color)

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

plt.show()

# plt.savefig(f"{path}kmeans_cluster_stats_bar_scatter.png", dpi=300, 
#             transparent=True, bbox_inches='tight')



