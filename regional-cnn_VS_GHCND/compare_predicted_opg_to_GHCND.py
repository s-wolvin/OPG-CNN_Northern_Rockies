"""
Savanna Wolvin
Created: Jul 31st, 2023
Edited: Jan 11th, 2024


##### SUMMARY #################################################################
This python script is designed to formulate the errors in predicted 
precipitation to the observed precipitation. Predicted precipitation is 
formulated from the true OPG values, the predicted training OPG values, and the 
predicted testing OPG values. The observed precipitation is the Global 
Historical Climatological Network (GHCN) - Daily data.

##### INPUT ###################################################################
fi_dir      - Directory to facet data
opg_dir     - Directory to OPG data
model_dir   - Directory to CNN model
model_name  - Name of Model Run
d_dir       - Directory to input data to model
years       - Years to evaluate OPG
months      - Months to evaluate OPG


##### OUTPUT ##################################################################
{path}map_{dataset}_{control}_vs_all_heatmap.png



"""
#%% Global Imports

import scipy.io as sio
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from tqdm import tqdm
import sys
sys.path.insert(1, '/uufs/chpc.utah.edu/common/home/u1324060/nclcmappy/')
import nclcmaps as ncm



#%%% Variable Presets

fi_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/facets/"
opg_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/opg/"

model_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/regional_facet_cnn_weighting/NR/"
model_name = "2024-01-08_1113"

d_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/ecmwf_era5/"
   
years = [1979, 2018]
months  = [12, 1, 2]



#%% Load in Facet Data

print('Load Facet Data...')

mat_file = sio.loadmat(fi_dir + 'lats')
lats  = mat_file['lats']

mat_file = sio.loadmat(fi_dir + 'lons')
lons  = mat_file['lons']

mat_file = sio.loadmat(fi_dir + 'elev')
elev  = mat_file['elev']

mat_file = sio.loadmat(fi_dir + 'facets_labeled')
facets  = mat_file['facets_i']

mat_file = sio.loadmat(opg_dir + 'allOPG_qc2')
act_opg  = mat_file['allOPG_qc2']


#%% Load in Observational Precipitation Data

print('Load Precipitation Data...')

mat_file = sio.loadmat(opg_dir + 'prcpout_15629')
ghcnd_pr  = mat_file['prcpout_15629'][:,3:]

mat_file = sio.loadmat(opg_dir + 'prcp_elev_15629')
ghcnd_elev  = mat_file['prcp_elev_15629']

mat_file = sio.loadmat(opg_dir + 'prcp_latlon_15629')
ghcnd_latlon  = mat_file['prcp_latlon_15629']

mat_file = sio.loadmat(fi_dir + 'station_facet_assignment')
ghcnd_assgnmnt  = mat_file['station_facet']

mat_file = sio.loadmat(opg_dir + 'newtest_2stn_meanp_20')
mean_pr_facets = mat_file['meanp']

mat_file = sio.loadmat(opg_dir + 'newtest_2stn_opg_int_20')
yInt = mat_file['allOPGint']


#%% Load Testing OPG Values and Precipitation from ERA5

print('Load CNN OPG and ERA5 Precipitation...')

# Load Output Stats
output = xr.open_dataset(f"{model_dir}{model_name}/stats/Testing_output_stats.nc", engine='netcdf4')
test_opg = output.predicted.values
nr_facets = output.facet_num.values
test_time = output.time.values

output = xr.open_dataset(f"{model_dir}{model_name}/stats/Training_output_stats.nc", engine='netcdf4')
train_opg = output.predicted.values
train_time = output.time.values

# Create time dataset
ghcnd_days = np.arange(datetime(years[0],1,1), datetime(years[1],4,1), timedelta(days=1)).astype(datetime)

# Load precipitation
era5_pr = xr.Dataset()
for data_yearsX in np.arange(years[0], years[1]+1): # loop through each year 
    # Access the NC File and Convert Dataset to an Xarray
    ncfile = xr.open_dataset(
        d_dir + "daily/sfc/era5_precip_" + str(data_yearsX) + "_oct-apr_daily.nc")
    
    # Save Atmospheric Variable
    era5_pr = xr.merge([era5_pr, ncfile])



#%% Formulate Mean Precipitation on the facets

print('Formulate Mean Precipitation...')

era5_mean_pr_test = np.zeros(np.shape(test_opg))
era5_mean_pr_train = np.zeros(np.shape(train_opg))

# Pull true precipitation mean
# Test
for dayx in tqdm(range(len(test_time))):
    # Pull date
    ghcnd_day = test_time[dayx] == ghcnd_days.astype('datetime64')
    
    for fi in range(len(nr_facets)):
        # same elevation average\
        elevx = [np.max(elev[facets==nr_facets[fi]]), np.min(elev[facets==nr_facets[fi]])]
        xxx = (act_opg[ghcnd_day, nr_facets[fi]-1] * elevx) +  \
                                                yInt[ghcnd_day, nr_facets[fi]-1]
        era5_mean_pr_test[dayx ,fi] = np.nanmean([np.nanmin(xxx), np.nanmax(xxx)])
        


# Train
for dayx in tqdm(range(len(train_time))):
    # Pull date
    ghcnd_day = train_time[dayx] == ghcnd_days.astype('datetime64')
    
    for fi in range(len(nr_facets)):
        # same elevation average
        elevx = [np.max(elev[facets==nr_facets[fi]]), np.min(elev[facets==nr_facets[fi]])]
        xxx = (act_opg[ghcnd_day, nr_facets[fi]-1] * elevx) +  \
                                                yInt[ghcnd_day, nr_facets[fi]-1]
        era5_mean_pr_train[dayx ,fi] = np.nanmean([np.nanmin(xxx), np.nanmax(xxx)])
        


#%% Formulate y-Intercept

print('Formulate y-Intercept...')

opg_yInt_test = np.zeros(np.shape(test_opg))
opg_yInt_train = np.zeros(np.shape(train_opg))

# Test
for timex in tqdm(range(len(test_time))):
    for fi in range(len(nr_facets)):
        mean_elev = np.mean([np.max(elev[facets==nr_facets[fi]]), np.min(elev[facets==nr_facets[fi]])])
        opg_yInt_test[timex, fi] = era5_mean_pr_test[timex ,fi] - \
                (test_opg[timex ,fi] * mean_elev)


# Train
for timex in tqdm(range(len(train_time))):
    for fi in range(len(nr_facets)):
        mean_elev = np.mean([np.max(elev[facets==nr_facets[fi]]), np.min(elev[facets==nr_facets[fi]])])
        opg_yInt_train[timex, fi] = era5_mean_pr_train[timex ,fi] - \
                (train_opg[timex ,fi] * mean_elev)



#%% Calculate precipitation

print('Calculate Precipitation at Stations...')
  
opg_pr = np.zeros((len(ghcnd_elev), len(ghcnd_days)))
opg_pr[:,:] = np.nan
  
cnn_pr_test = np.zeros((len(ghcnd_elev), len(test_time)))
cnn_pr_test[:,:] = np.nan

cnn_pr_train = np.zeros((len(ghcnd_elev), len(train_time)))
cnn_pr_train[:,:] = np.nan

ghcnd_assgnmnt[np.isnan(ghcnd_assgnmnt)] = 0

# Actual OPG
for fi in nr_facets-1:
    for stx in range(len(ghcnd_elev)):
        if int(ghcnd_assgnmnt[stx]) == fi+1:
            opg_pr[stx, :] = (act_opg[:, fi] * ghcnd_elev[stx]) + yInt[:, fi]

# Test
for fi in range(len(nr_facets)):
    for stx in range(len(ghcnd_elev)):
        if int(ghcnd_assgnmnt[stx]) == nr_facets[fi]:
            cnn_pr_test[stx, :] = (test_opg[:, fi] * ghcnd_elev[stx]) + opg_yInt_test[:, fi]
                
# Train
for fi in range(len(nr_facets)):
    for stx in range(len(ghcnd_elev)):
        if int(ghcnd_assgnmnt[stx]) == nr_facets[fi]:
            cnn_pr_train[stx, :] = (train_opg[:, fi] * ghcnd_elev[stx]) + opg_yInt_train[:, fi]


#%% Formulate the error

print('Calculate Precipitation Error...')

opg_pr[opg_pr < 0] = 0
cnn_pr_test[cnn_pr_test < 0] = 0
cnn_pr_train[cnn_pr_train < 0] = 0
ghcnd_pr[ghcnd_pr == 0] = np.nan # Set no precip days as nans
# ghcnd_pr[np.isnan(ghcnd_pr)] = 0

e_pr_opg = np.zeros((len(ghcnd_elev), len(ghcnd_days)))
e_pr_test = np.zeros((len(ghcnd_elev), len(test_time)))
e_pr_train = np.zeros((len(ghcnd_elev), len(train_time)))

ae_pr_opg = np.zeros((len(ghcnd_elev), len(ghcnd_days)))
ae_pr_test = np.zeros((len(ghcnd_elev), len(test_time)))
ae_pr_train = np.zeros((len(ghcnd_elev), len(train_time)))

re_pr_opg = np.zeros((len(ghcnd_elev), len(ghcnd_days)))
re_pr_test = np.zeros((len(ghcnd_elev), len(test_time)))
re_pr_train = np.zeros((len(ghcnd_elev), len(train_time)))

e_opg_train = np.zeros((len(ghcnd_elev), len(train_time)))
e_opg_test = np.zeros((len(ghcnd_elev), len(test_time)))

ae_opg_train = np.zeros((len(ghcnd_elev), len(train_time)))
ae_opg_test = np.zeros((len(ghcnd_elev), len(test_time)))

re_opg_train = np.zeros((len(ghcnd_elev), len(train_time)))
re_opg_test = np.zeros((len(ghcnd_elev), len(test_time)))

# Actual
for timex in tqdm(range(len(ghcnd_days))):
    idx = ghcnd_days[timex] == ghcnd_days
    e_pr_opg[:, timex] = opg_pr[:, timex] - ghcnd_pr[idx, :] 
    ae_pr_opg[:, timex] = np.abs(opg_pr[:, timex] - ghcnd_pr[idx, :])
    re_pr_opg[:, timex] = (np.abs(opg_pr[:, timex] - ghcnd_pr[idx, :]) / ghcnd_pr[idx, :]) * 100

# Test
for timex in tqdm(range(len(test_time))):
    idx = test_time[timex] == ghcnd_days.astype('datetime64[ns]')
    
    # test VS ghcnd
    e_pr_test[:, timex] = cnn_pr_test[:, timex] - ghcnd_pr[idx, :]
    ae_pr_test[:, timex] = np.abs(cnn_pr_test[:, timex] - ghcnd_pr[idx, :])
    re_pr_test[:, timex] = (np.abs(cnn_pr_test[:, timex] - ghcnd_pr[idx, :]) / ghcnd_pr[idx, :]) * 100
    
    # Test vs OPG
    e_opg_test[:, timex] = cnn_pr_test[:, timex] - opg_pr[:, idx].T
    ae_opg_test[:, timex] = np.abs(cnn_pr_test[:, timex] - opg_pr[:, idx].T)
    re_opg_test[:, timex] = (np.abs(cnn_pr_test[:, timex] - opg_pr[:, idx].T) / opg_pr[:, idx].T) * 100
    

# Train
for timex in tqdm(range(len(train_time))):
    idx = train_time[timex] == ghcnd_days.astype('datetime64[ns]')
    
    # train vs ghcnd
    e_pr_train[:, timex] = cnn_pr_train[:, timex] - ghcnd_pr[idx, :]
    ae_pr_train[:, timex] = np.abs(cnn_pr_train[:, timex] - ghcnd_pr[idx, :])
    re_pr_train[:, timex] = (np.abs(cnn_pr_train[:, timex] - ghcnd_pr[idx, :]) / ghcnd_pr[idx, :]) * 100
    
    # train vs OPG
    e_opg_train[:, timex] = cnn_pr_train[:, timex] - opg_pr[:, idx].T
    ae_opg_train[:, timex] = np.abs(cnn_pr_train[:, timex] - opg_pr[:, idx].T)
    re_opg_train[:, timex] = (np.abs(cnn_pr_train[:, timex] - opg_pr[:, idx].T) / opg_pr[:, idx].T) * 100
    
    
#%% Remove facets not in the domain

print('Remove Observations Outside of Domain...')

# Actual
for stx in tqdm(range(len(ghcnd_elev))):
    if np.sum(ghcnd_assgnmnt[stx] == nr_facets) == 0:
        e_pr_opg[stx,:] = np.nan
        ae_pr_opg[stx,:] = np.nan
        re_pr_opg[stx,:] = np.nan
        
        e_pr_test[stx,:] = np.nan
        ae_pr_test[stx,:] = np.nan
        re_pr_test[stx,:] = np.nan
        
        e_pr_train[stx,:] = np.nan
        ae_pr_train[stx,:] = np.nan
        re_pr_train[stx,:] = np.nan
        
        e_opg_test[stx,:] = np.nan
        ae_opg_test[stx,:] = np.nan
        re_opg_test[stx,:] = np.nan
        
        e_opg_train[stx,:] = np.nan
        ae_opg_train[stx,:] = np.nan
        re_opg_train[stx,:] = np.nan
    

me_pr_opg = np.nanmean(e_pr_opg, axis=1)
mae_pr_opg = np.nanmean(ae_pr_opg, axis=1)
mre_pr_opg = np.nanmean(re_pr_opg, axis=1)

me_pr_test = np.nanmean(e_pr_test, axis=1)
mae_pr_test = np.nanmean(ae_pr_test, axis=1)
mre_pr_test = np.nanmean(re_pr_test, axis=1)

me_pr_train = np.nanmean(e_pr_train, axis=1)
mae_pr_train = np.nanmean(ae_pr_train, axis=1)
mre_pr_train = np.nanmean(re_pr_train, axis=1)

me_opg_test = np.nanmean(e_opg_test, axis=1)
mae_opg_test = np.nanmean(ae_opg_test, axis=1)
mre_opg_test = np.nanmean(re_opg_test, axis=1)

me_opg_train = np.nanmean(e_opg_train, axis=1)
mae_opg_train = np.nanmean(ae_opg_train, axis=1)
mre_opg_train = np.nanmean(re_opg_train, axis=1)


#%% Voronoi

df = pd.DataFrame({
        "me_pr_opg": me_pr_opg, "mae_pr_opg": mae_pr_opg, "mre_pr_opg": mre_pr_opg,
        "me_pr_test": me_pr_test, "mae_pr_test": mae_pr_test, "mre_pr_test": mre_pr_test,
        "me_pr_train": me_pr_train, "mae_pr_train": mae_pr_train, "mre_pr_train": mre_pr_train,
        "me_opg_test": me_opg_test, "mae_opg_test": mae_opg_test, "mre_opg_test": mre_opg_test,
        "me_opg_train": me_opg_train, "mae_opg_train": mae_opg_train, "mre_opg_train": mre_opg_train,
        "lats": ghcnd_latlon[:,0], "lons": ghcnd_latlon[:,1]
    }
    )



#%% aggregated into heatmap

def agg_heatmap(dfx, stats, binsx, stepx, bin_step):
    # Remove rows with nan of stats
    dfx = dfx.dropna(subset=stats)
    
    # bin the lats and lons
    # dfx['lats'] = pd.cut(dfx['lats'], bins=np.arange(39, 51.1, 0.1), 
    #                       labels=np.arange(39, 51.05, 0.1))
    # dfx['lons'] = pd.cut(dfx['lons'], bins=np.arange(-118, -102, 0.1), 
    #                       labels=np.arange(-118, -102.15, 0.1))
    
    dfx['lats'] = pd.cut(dfx['lats'], bins=np.arange(39, 51.25, 0.25), 
                          labels=np.arange(39, 51, 0.25))
    dfx['lons'] = pd.cut(dfx['lons'], bins=np.arange(-118, -102.25, 0.25), 
                          labels=np.arange(-118, -102.5, 0.25))
    
    # dfx['lats'] = pd.cut(dfx['lats'], bins=np.arange(39, 51.5, 0.5), 
    #                       labels=np.arange(39, 51, 0.5))
    # dfx['lons'] = pd.cut(dfx['lons'], bins=np.arange(-118, -102.5, 0.5), 
    #                       labels=np.arange(-118, -103, 0.5))
    
    # Group and count
    dfx = dfx.groupby(['lats','lons']).mean().reset_index()
    
    return dfx


# data presets
dataset = 'me'
control = 'pr'
bins    = [-3,-2,-1,0,1,2,3]#[-2, -1, 0, 1, 2]
bin_step = 0.5
lbls    = [-2.5,-1.5,-0.5,0.5,1.5,2.5]#[-1.5, -0.5, 0.5, 1.5]
vval    = 4
mx_val  = 125
wx        = 1
leg     = ["$\leq$-3", "(-3, -2]", "(-2, -1]","(-1, 0]","(0, 1]","(1, 2]","(2, 3]",">3"] #["$\leq$-2", "(-2, -1]","(-1, 0]","(0, 1]","(1, 2]",">2"]
l_title = 'Mean Precipitation Error (mm)'


var     = ['opg', 'train', 'test']
titlex  = ['a) Actual OPG', 'b) Training OPG', 'c) Testing OPG']

# map presets
bounds  = [40.5, 49.5, -116, -105]
bbounds = [bounds[0]-0.75, bounds[1]+0.25, bounds[2]-1, bounds[3]]
datacrs = ccrs.PlateCarree()
cmap    = ncm.cmap('MPL_gist_gray')
ccmap   = ncm.cmapDiscrete('BlueDarkRed18', indexList=np.round(np.linspace(0,17, num=8)).astype('int'))#ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[32,44,81,111,146,168])
tt_sz   = 14
lbl_sz  = 12
bkgnd = 'gray'


# make plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 6),
                        subplot_kw={'projection': ccrs.Mercator(central_longitude=np.mean(lons))})

###################################
# add heatmap
dfx = agg_heatmap(df[['lons', 'lats', f'{dataset}_{control}_opg']], 
                           f'{dataset}_{control}_opg', bins.copy(), lbls.copy(), bin_step)
X, Y = np.meshgrid(np.unique(dfx['lons'].values), np.unique(dfx['lats'].values))
dfx = dfx.pivot(index='lats', columns='lons', values='me_pr_opg')    
ax[0].pcolormesh(X, Y, dfx.values, transform=datacrs, cmap=ccmap, vmin=-vval, vmax=vval, zorder=3)

# Cartography
ax[0].add_feature(cfeat.LAND, facecolor=bkgnd)
ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", linewidth=2, zorder=2)
ax[0].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", linewidth=2, zorder=1)
ax[0].set_extent([bbounds[2], bbounds[3], bbounds[0], bbounds[1]])

# set title
ax[0].set_title(titlex[0], fontsize=tt_sz)

########################################
# add heatmap
dfx = agg_heatmap(df[['lons', 'lats', f'{dataset}_{control}_train']], 
                           f'{dataset}_{control}_train', bins.copy(), lbls.copy(), bin_step)
X, Y = np.meshgrid(np.unique(dfx['lons'].values), np.unique(dfx['lats'].values))
dfx = dfx.pivot(index='lats', columns='lons', values='me_pr_train')    
ax[1].pcolormesh(X, Y, dfx.values, transform=datacrs, cmap=ccmap, vmin=-vval, vmax=vval, zorder=3)

# Cartography
ax[1].add_feature(cfeat.LAND, facecolor=bkgnd)
ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", linewidth=2, zorder=2)
ax[1].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", linewidth=2, zorder=1)
ax[1].set_extent([bbounds[2], bbounds[3], bbounds[0], bbounds[1]])

# set title
ax[1].set_title(titlex[1], fontsize=tt_sz)

##########################################
# add heatmap
dfx = agg_heatmap(df[['lons', 'lats', f'{dataset}_{control}_test']], 
                           f'{dataset}_{control}_test', bins.copy(), lbls.copy(), bin_step)
X, Y = np.meshgrid(np.unique(dfx['lons'].values), np.unique(dfx['lats'].values))
dfx = dfx.pivot(index='lats', columns='lons', values='me_pr_test')    
pcm = ax[2].pcolormesh(X, Y, dfx.values, transform=datacrs, cmap=ccmap, vmin=-vval, vmax=vval, zorder=3)

# Cartography
ax[2].add_feature(cfeat.LAND, facecolor=bkgnd)
ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", linewidth=2, zorder=2)
ax[2].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", linewidth=2, zorder=1)
ax[2].set_extent([bbounds[2], bbounds[3], bbounds[0], bbounds[1]])

# set title
ax[2].set_title(titlex[2], fontsize=tt_sz)




# Add colorbar
cbar_opg = fig.add_axes([0.263, 0.18, 0.5, 0.04]) #[x0, y0, width, height]
cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6],#ticks=[-10, -8, -6, -4, -2, 0],
                      extend='both', orientation='horizontal', pad=0.0, 
                      aspect=5, fraction=0.032)
cb_opg.set_label('Mean Precipitation Error (mm)', size=lbl_sz)
cb_opg.ax.tick_params(labelsize=lbl_sz)

plt.subplots_adjust(wspace=0.05, hspace=0.05)




# Save and Show Figure
path = model_dir + model_name + "/"
plt.savefig(f"{path}map_{dataset}_{control}_vs_all_heatmap.png", dpi=300, 
            transparent=True, bbox_inches='tight')

plt.show()
plt.clf()
