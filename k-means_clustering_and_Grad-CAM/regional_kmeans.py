"""
Savanna Wolvin
Created: May 15th, 2023
Edited: May 23rd, 2024

##### SUMMARY ################################################################
This script takes the OPG values from the specified region and formulates 
K-means clusters in the goal of dividing the study domain into subsets of OPG 
events for analysis of the CNN.


##### INPUT ##################################################################
opg_dir     - Directory to the OPG values
fi_dir      - Directory to the Facet data
kmeans_dir  - Directory to the Kmeans analysis
years       - Years to pull
months      - Months to pull
num_station - Number of stations required for the OPG to be valid
fi_region   - Region of focus
opg_nans    - How to Handle NaN OPGs
opg_type    - Type of OPGs used in training
prct_obs    - Percent of observations for a facet to be valid
num_station - Minimum number of stations for an OPG observation to be valid


##### OUTPUT #################################################################



"""
#%% Global Imports

import scipy.io as sio
import xarray as xr
import numpy as np
import scipy.stats as sp_stats
from datetime import timedelta, datetime
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat




#%% Variable Presets

opg_dir     = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/opg/"
fi_dir      = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/facets/"
kmeans_dir  = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_kmeans_clust/"


years           = [1979, 2018]
months          = [12, 1, 2]
num_station     = 3

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




#%% Load lats/lons/facets/orientation
mat_file = sio.loadmat(fi_dir + 'lats')
lats  = mat_file['lats']

mat_file = sio.loadmat(fi_dir + 'lons')
lons  = mat_file['lons']

mat_file = sio.loadmat(fi_dir + 'facets')
orientation  = mat_file['facets']

mat_file = sio.loadmat(fi_dir + 'facets_labeled')
facets  = mat_file['facets_i']




#%% Pull OPG Values

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




#%% K-Means Clustering

K = 15
sse = np.zeros((K,1))

for k in range(K):
    # Cluster the dataset
    kmeans = KMeans(k+1, n_init='auto', random_state=0)
    kmeans = kmeans.fit(facet_opg.opg.values)
    sse[k] = kmeans.inertia_

k_vals = np.expand_dims(np.arange(1,k+2), axis=1)



#%% Plot SSE of Kmeans clustering

plt.figure(figsize=(7,5))

plt.plot(range(1,K+1), np.squeeze(sse), c='blue')
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Error")
plt.xlim([0, K])
plt.ylim([2.5,6])
plt.grid(True)
plt.rcParams.update({'font.size': 16})

# plt.savefig(f"{kmeans_dir}opg_{fi_region}_kmeans_sse.png", dpi=400, transparent=True, \
#             bbox_inches='tight')

plt.show()




#%% Plot the clusters of OPG values from set elbow over the study domain

set_elbow = 5

kmeans = KMeans(set_elbow, n_init='auto', random_state=0)
kmeans_clust = kmeans.fit_predict(facet_opg.opg.values)

for clust in range(set_elbow):
    
    # Take Mean of OPGs by Cluster
    opgs = facet_opg.opg.values.copy()
    opgs[opgs==0] = np.nan
    clust_opg = np.nanmean(opgs[kmeans_clust==clust,:], axis=0)
    
    # Create Matrix
    opgs = np.zeros(np.shape(facets))
    opgs[:] = np.nan
    for fi in range(len(facet_opg.facet_num)):
        opgs[facets==facet_opg.facet_num[fi].values] = clust_opg[fi]
    
    # Create Figure
    datacrs = ccrs.PlateCarree()
    fig = plt.figure( figsize = (6, 8))
    ax = fig.add_axes( [0.1, 0.1, 0.8, 0.8], 
                      projection = ccrs.Mercator(central_longitude=np.mean(lons)))
    
    # Add OPG by Facet
    ccmap = plt.get_cmap('bwr', 24)
    pcm = ax.pcolormesh(lons, lats, opgs, cmap=ccmap, transform=datacrs, shading='auto')
    pcm.set_clim(-0.015,0.015)
    
    # Add Regional Box
    plt.plot([bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]],
              [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]],
              transform=datacrs, color='black')
    
    # Cartography
    ax.add_feature(cfeat.LAND, facecolor="burlywood")
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="saddlebrown")
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="saddlebrown")
    ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="saddlebrown")
    ax.set_extent([bounds[2]-1, bounds[3]+1, bounds[0]-1, bounds[1]+1])

    # Add colorbar
    cbar = plt.colorbar(pcm, pad=0.01, extend='both', location='bottom', 
                        ticks=[-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015])
    cbar.set_label("Mean OPG", fontsize=14)
    cbar.set_ticklabels(['', '-0.01', '', '0.0', '', '0.01', ''])
    
    # Add Title
    ax.set_title("Cluster " + str(clust+1))

    # Save and Show Figure
    print("Save Figure of Labeled Facets...")
    # plt.savefig(f"{kmeans_dir}kmeans_cluster_{str(clust+1)}_{fi_region}.png", dpi=400, transparent=True, \
    #             bbox_inches='tight')

    plt.show()
    
    
    
#%% Save the Cluster output

df = pd.DataFrame((facet_opg.time.values), columns=['datetime'])
df['cluster'] = kmeans_clust
df['mean_opg'] = np.nanmean(facet_opg.opg.values, axis=1)
df['std_opg'] = np.nanstd(facet_opg.opg.values, axis=1)
df.to_csv(f"{kmeans_dir}kmeans_clusters_opg_{fi_region}")
    


#%% Formulate the Mean and STD of each cluster

opg_mean = np.zeros((set_elbow,1))
opg_std = np.zeros((set_elbow,1))
opg_count = np.zeros((set_elbow,1))

for dx in range(len(kmeans_clust)):
    opg_count[kmeans_clust[dx]] += 1
    opg_mean[kmeans_clust[dx]] += np.nanmean(facet_opg.opg.values[dx,:])
    opg_std[kmeans_clust[dx]] += np.nanstd(facet_opg.opg.values[dx,:])
    

opg_mean /= opg_count
opg_std /= opg_count


#%% define colormap

cmapx = [[0,	0,	0],
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

cmap = [ccmap[i] for i in np.array((np.round(np.linspace(0, len(cmapx)-1, num=set_elbow))), dtype='int')]


ccmap = plt.get_cmap('nipy_spectral', set_elbow)

#%% Plot Values of mean and std of opg on these days

fig = plt.figure( figsize = (6,5))

plt.scatter(opg_mean, opg_std, s=150, c=range(1,set_elbow+1), cmap=ccmap, edgecolors='black')

plt.xlabel("OPG Mean")
plt.ylabel("OPG Standard Deviation")

plt.xlim([0,0.012])
plt.ylim([0,0.012])

plt.xticks([0.0, 0.005, 0.01])
plt.yticks([0.0, 0.005, 0.01])

plt.colorbar(ticks=range(1,set_elbow+1), label='Cluster Number')
plt.clim(0.5,set_elbow+0.5)
plt.grid(True)

# plt.savefig(f"{kmeans_dir}kmeans_clusters_{fi_region}_meanSTD.png", dpi=400, transparent=True, \
#             bbox_inches='tight')

plt.show()



#%% Plot Timeseries of Clusters

plt.figure(figsize=(15,3))

for clust in range(set_elbow):
    
    cx_idx = (kmeans_clust != clust)
    cx = kmeans_clust.copy().astype('float') + 1
    cx[cx_idx] = np.nan
    
    plt.scatter(range(len(facet_opg.time.values)), cx, marker='+', linewidths=2, c=cmap[clust])


plt.yticks(np.arange(1,set_elbow+1))#[1,2,3,4,5,6,7,8,9,10,11])
plt.ylabel('Cluster')
plt.xlim([0,3579])
plt.xlabel('Time')
plt.xticks([0, 451, 903, 1354, 1805, 2256, 2708, 3159], 
           [1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014])

# plt.savefig(f"{kmeans_dir}kmeans_clusters_{fi_region}_timesries.png", dpi=400, transparent=True, \
#             bbox_inches='tight')

plt.show()






#%% Plot Count per year

dates = pd.to_datetime(facet_opg.time.values)

plt.figure(figsize=(15,6))

for clust in range(set_elbow):
    
    cx_idx = (kmeans_clust != clust)
    cx = kmeans_clust.copy().astype('float') + 1
    cx[cx_idx] = np.nan
    
    counts = []
    for yx in d_years:
        idx = (dates > pd.to_datetime(f'{yx-1}1130', format='%Y%m%d')) & (dates < pd.to_datetime(f'{yx}0401', format='%Y%m%d'))
        counts.append(np.sum(~cx_idx[idx]))
        
    plt.plot(d_years, counts, c=cmap[clust])


plt.ylabel("Year")
plt.xlabel("Count of Events per Year")
plt.grid(True)
plt.show()

