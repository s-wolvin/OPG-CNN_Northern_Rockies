"""
Savanna Wolvin
Created: Jul 31st, 2023
Edited: Oct 4th, 2023


##### SUMMARY #################################################################
This python script is designed to formulate the errors in predicted 
precipitation to the observed precipitation. Predicted precipitation is 
formulated from the true OPG values, the predicted training OPG values, and the 
predicted testing OPG values. The observed precipitation is the Global 
Historical Climatological Network (GHCN) - Daily data.

##### INPUT ###################################################################


##### OUTPUT ##################################################################


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
        
        # same areal average
        # era5_mean_pr_test[dayx ,fi] = mean_pr_facets[ghcnd_day, nr_facets[fi]-1]


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
        
        # same areal average
        # era5_mean_pr_train[dayx ,fi] = mean_pr_facets[ghcnd_day, nr_facets[fi]-1]
        


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
        
        
        # This mean might be skewing the values because of the distribution of elevation points
        # opg_yInt_test[timex, fi] = era5_mean_pr_test[timex ,fi] - \
        #         (test_opg[timex ,fi] * np.mean(elev[facets==nr_facets[fi]]))

# Train
for timex in tqdm(range(len(train_time))):
    for fi in range(len(nr_facets)):
        mean_elev = np.mean([np.max(elev[facets==nr_facets[fi]]), np.min(elev[facets==nr_facets[fi]])])
        opg_yInt_train[timex, fi] = era5_mean_pr_train[timex ,fi] - \
                (train_opg[timex ,fi] * mean_elev)
                
                
        # This mean might be skewing the values because of the distribution of elevation points
        # opg_yInt_train[timex, fi] = era5_mean_pr_train[timex ,fi] - \
        #         (train_opg[timex ,fi] * np.mean(elev[facets==nr_facets[fi]]))
                



#%% plot the OPGs to compare

# training days

# for dayx in range(len(test_time)):
#     # Pull date
#     ghcnd_day = test_time[dayx] == ghcnd_days.astype('datetime64')
    
#     for fi in range(len(nr_facets)):
        
#         # Plot GHCND
#         x_gh = ghcnd_elev[ghcnd_assgnmnt==nr_facets[fi]]
#         y_gh = ghcnd_pr[ghcnd_day, np.squeeze(ghcnd_assgnmnt==nr_facets[fi])]
        
#         if not np.isnan(act_opg[ghcnd_day, nr_facets[fi]-1]):
#             plt.scatter(x_gh, y_gh, c='k')
            
#             # Plot Formulated OPG
#             x = np.linspace(np.min(elev[facets==nr_facets[fi]]), np.max(elev[facets==nr_facets[fi]]), num=10)
#             y_act = (act_opg[ghcnd_day, nr_facets[fi]-1]) * x + yInt[ghcnd_day, nr_facets[fi]-1]
#             plt.plot(x, y_act)
            
#             # Plot CNN OPG
#             y_pre = (test_opg[dayx, fi] * x) + opg_yInt_test[dayx, fi]
#             plt.plot(x, y_pre)
            
#             plt.grid()
#             plt.title(str(nr_facets[fi]) + ': ' + test_time[dayx].astype('str')[0:10])
            
#             plt.legend(['GHCND-Daily', 
#                         'Actual: ' + str(np.round(mean_pr_facets[ghcnd_day, nr_facets[fi]-1][0], decimals=3)), 
#                         'Pred: ' + str(np.round(era5_mean_pr_test[dayx ,fi], decimals=3))])
            
#             plt.ylim((-10,50))
#             plt.xlim((500, 4500))
            
            
#             plt.show()
#             plt.clf()
        
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
# for timex in tqdm(range(len(ghcnd_days))):
for fi in nr_facets-1:
    for stx in range(len(ghcnd_elev)):
        if int(ghcnd_assgnmnt[stx]) == fi+1:
            opg_pr[stx, :] = (act_opg[:, fi] * ghcnd_elev[stx]) + yInt[:, fi]

# Test
# for timex in tqdm(range(len(test_time))):
for fi in range(len(nr_facets)):
    for stx in range(len(ghcnd_elev)):
        if int(ghcnd_assgnmnt[stx]) == nr_facets[fi]:
            cnn_pr_test[stx, :] = (test_opg[:, fi] * ghcnd_elev[stx]) + opg_yInt_test[:, fi]
                
# Train
# for timex in tqdm(range(len(train_time))):
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



#%% Pir chart scatter plot

# # data presets
# bins = [0,1,2,3,4,5]
# lbls = [0.5,1.5,2.5,3.5,4.5]

# # map presets
# bounds = [40.5, 49.5, -116, -105]
# bbounds = [bounds[0]-0.75, bounds[1]+0.75, bounds[2]-0.75, bounds[3]+0.75]
# datacrs = ccrs.PlateCarree()
# cmap = ncm.cmap('MPL_gist_gray')
# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[28,44,81,111,146,176])


# def formulate_bar_values(dfx, stats, bins, step):
#     # Remove rows with nan of stats
#     dfx = dfx.dropna(subset=stats)
    
#     # bin the lats and lons
#     dfx['lats'] = pd.cut(dfx['lats'], bins=[40, 42, 44, 46, 48, 50], labels=[41, 43, 45, 47, 49])
#     dfx['lons'] = pd.cut(dfx['lons'], bins=[-116.5, -113.5, -110.5, -107.5, -104.5], labels=[-115, -112, -109, -106])
    
#     # Create Bins
#     if np.nanmin(dfx[stats]) < np.nanmin(bins): 
#         bins.insert(0, np.nanmin(dfx[stats]))
#         step.insert(0, bins[0]-0.5)
#     if np.nanmax(dfx[stats]) > np.nanmax(bins): 
#         bins.append(np.nanmax(dfx[stats]))
#         step.append(bins[-2]+0.5)
    
#     # Bin the stats data
#     dfx['color'] = pd.cut(dfx[stats], bins=bins, labels=step)
        
#     # Group and count
#     dfx = dfx.groupby(['lats','lons', 'color']).size().reset_index().rename(columns={0:'size'})
    
#     return dfx, len(step)

# # preset values
# df2, step = formulate_bar_values(df[['lons', 'lats', 'mae_pr_train']], 'mae_pr_train', bins, lbls)

# # make plot
# fig = plt.figure(figsize=(5,6))
# ax = fig.add_axes( [0, 0, 1, 1], projection = ccrs.Mercator(central_longitude=np.mean(lons)))


# # add background terrain
# pcm = ax.contourf(lons[0:359:10,263:672:10], lats[0:359:10,263:672:10], elev[0:359:10,263:672:10], 
#                      cmap=cmap, transform=datacrs, levels=range(0,4500,100), 
#                      extend='upper', zorder=1)

# # add bar chart
# for idx in range(0, df2.shape[0], step):
#     lat = df2.iloc[idx, 0]
#     lon = df2.iloc[idx, 1]
#     color = df2.iloc[idx:idx+step, 2]
#     size = df2.iloc[idx:idx+step, 3]
    
#     if np.sum(size) > 0:
#         ax_bar = fig.add_axes([(lon-(bbounds[2]))/((bbounds[3])-(bbounds[2]))-0.11 , 
#                                (lat-(bbounds[0]))/((bbounds[1])-(bbounds[0]))-0.1 , 
#                                0.22, 0.18])
#         pcm = ax_bar.bar(color, size, color=ccmap.colors, width=1, edgecolor=[0,0,0])
#         ax_bar.set_ylim([0, np.max(df2['size'])*1.1])
        
#         ax_bar.patch.set_alpha(0.01) # set background to transparent
#         ax_bar.set_xticks([]) # remove tick labels
#         ax_bar.set_yticks([]) # remove tick labels
#         ax_bar.spines['top'].set_visible(False) # remove top line
#         ax_bar.spines['right'].set_visible(False) # remove right line
#         # ax_bar.spines['left'].set_visible(False) # remove left line
#         #ax_bar.set_axis_off()


# ##### ADD LINES TO PLOT TO FIND WHAT YOU ARE EVEN DOING
# ax.scatter(df2.iloc[:, 1], df2.iloc[:, 0], color='red', transform=datacrs)
# #####

# # Cartography
# ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", linewidth=2, zorder=2)
# ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", linewidth=2, zorder=3)
# ax.set_extent([bbounds[2], bbounds[3], bbounds[0], bbounds[1]])

# ax.legend(pcm, ["<1","1-2","2-3","3-4","4-5",">5"], title='MAE', loc='upper right')

# plt.show()



#%% bar chart overlay for 3 plots

def form_bar_vals_4x5(dfx, stats, binsx, stepx, bin_step):
    # Remove rows with nan of stats
    dfx = dfx.dropna(subset=stats)
    
    # bin the lats and lons
    dfx['lats'] = pd.cut(dfx['lats'], bins=[40, 42, 44, 46, 48, 50], labels=[41, 43, 45, 47, 49])
    dfx['lons'] = pd.cut(dfx['lons'], bins=[-116.5, -113.5, -110.5, -107.5, -104.5], labels=[-115, -112, -109, -106])
    
    # Create Bins
    if np.nanmin(dfx[stats]) < np.nanmin(binsx): 
        binsx.insert(0, np.nanmin(dfx[stats]))
        stepx.insert(0, binsx[1]-bin_step)
    if np.nanmax(dfx[stats]) > np.nanmax(binsx): 
        binsx.append(np.nanmax(dfx[stats]))
        stepx.append(binsx[-2]+bin_step)
    
    # Bin the stats data
    dfx['color'] = pd.cut(dfx[stats], bins=binsx, labels=stepx)
        
    # Group and count
    dfx = dfx.groupby(['lats','lons', 'color']).size().reset_index().rename(columns={0:'size'})
    
    return dfx, len(stepx)


#%% bar chart a 4x5

# data presets
# dataset = 'mae'
# control = 'pr'
# bins    = [0,1,2,3,4,5]
# bin_step = 0.5
# lbls    = [0.5,1.5,2.5,3.5,4.5]
# mx_val  = 80
# wx        = 1
# leg     = ["<1","1$-$2","2$-$3","3$-$4","4$-$5",">5"]
# l_title = 'MAE (mm)'

# dataset = 'me'
# control = 'pr'
# bins    = [-2, -1, 0, 1, 2]
# bin_step = 0.5
# lbls    = [-1.5, -0.5, 0.5, 1.5]
# mx_val  = 80
# wx        = 1
# leg     = ["< -2", "-2 $-$ -1","-1 $-$ 0","0 $-$ 1","1 $-$ 2","> 2"]
# l_title = 'ME (mm)'

dataset = 'mre'
control = 'pr'
bins    = [0,50,100,150,200,250]
bin_step = 25
lbls    = [25,75,125,175,225]
mx_val  = 125
wx      = 50
leg     = ["0$-$50","50$-$100","100$-$150","150$-$200","200$-$250",">250"]
l_title = 'MRE (%)'

var     = ['opg', 'train', 'test']
titlex  = ['a) GHCND vs Observed OPG', 'b) GHCND vs Trained OPG', 'c) GHCND vs Testing OPG']

# map presets
bounds  = [40.5, 49.5, -116, -105]
bbounds = [bounds[0]-0.75, bounds[1]+0.75, bounds[2]-0.75, bounds[3]+0.75]
datacrs = ccrs.PlateCarree()
cmap    = ncm.cmap('MPL_gist_gray')
ccmap   = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[32,44,81,111,146,168])
tt_sz   = 22
lbl_sz  = 14


for cn in range(3):
    # pull variable
    dfx, step = form_bar_vals_4x5(df[['lons', 'lats', f'{dataset}_{control}_{var[cn]}']], 
                               f'{dataset}_{control}_{var[cn]}', bins.copy(), lbls.copy(), bin_step)

    # make plot
    fig = plt.figure(figsize=(5,6))
    ax = fig.add_axes( [0, 0, 1, 1], projection = ccrs.Mercator(
                                            central_longitude=np.mean(lons)))
    
    # add terrain
    ax.contourf(lons[0:359,263:672], lats[0:359,263:672], 
                elev[0:359,263:672], cmap=cmap, transform=datacrs, 
                levels=range(-500,4500,100), extend='upper', zorder=1)
    
    # add bar chart
    for idx in range(0, dfx.shape[0], step):
        lat = dfx.iloc[idx, 0]
        lon = dfx.iloc[idx, 1]
        
        if np.sum(dfx.iloc[idx:idx+step, 3]) > 0:
            ax_bar = fig.add_axes([(lon-(bbounds[2]))/((bbounds[3])-(bbounds[2]))-0.11 , 
                                   (lat-(bbounds[0]))/((bbounds[1])-(bbounds[0]))-0.1 , 
                                   0.22, 0.18])
            pcm = ax_bar.bar(dfx.iloc[idx:idx+step, 2], dfx.iloc[idx:idx+step, 3], 
                             color=ccmap.colors, width=wx, edgecolor=[0,0,0])
            
            ax_bar.set_ylim([0, mx_val])
            
            ax_bar.patch.set_alpha(0.01) # set background to transparent
            ax_bar.set_xticks([]) # remove tick labels
            ax_bar.set_yticks([]) # remove tick labels
            ax_bar.spines['top'].set_visible(False) # remove top line
            ax_bar.spines['right'].set_visible(False) # remove right line
            # ax_bar.spines['left'].set_visible(False) # remove left line
            #ax_bar.set_axis_off()
            
            
    ##### ADD LINES TO PLOT TO FIND WHAT YOU ARE EVEN DOING
    # ax.scatter(dfx.iloc[:, 1], dfx.iloc[:, 0], color='red', transform=datacrs)
    #####

    # Cartography
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", linewidth=2, zorder=2)
    ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", linewidth=2, zorder=3)
    ax.set_extent([bbounds[2], bbounds[3], bbounds[0], bbounds[1]])

    if cn == 0:
        legg = ax.legend(pcm, leg, loc='upper right', fontsize=lbl_sz)
        legg.set_title(l_title, prop={'size':lbl_sz})
    
    # set title
    ax.set_title(titlex[cn], fontsize=tt_sz)

    # Save and Show Figure
    path = model_dir + model_name + "/"
    plt.savefig(f"{path}map_{dataset}_{control}_vs_{var[cn]}.png", dpi=200, 
                transparent=True, bbox_inches='tight')
    
    plt.show()
    plt.clf()


#%% bar chart 3x4

def form_bar_vals_3x4(dfx, stats, binsx, stepx, bin_step):
    # Remove rows with nan of stats
    dfx = dfx.dropna(subset=stats)
    
    # bin the lats and lons
    dfx['lats'] = pd.cut(dfx['lats'], bins=[40, 42.5, 45, 47.5, 50], labels=[41.25, 43.75, 46.25, 48.75])
    dfx['lons'] = pd.cut(dfx['lons'], bins=[-116.5, -112.5, -108.5, -104.5], labels=[-114.5, -110.5, -106.5])
    
    # Create Bins
    if np.nanmin(dfx[stats]) < np.nanmin(binsx): 
        binsx.insert(0, np.nanmin(dfx[stats]))
        stepx.insert(0, binsx[1]-bin_step)
    if np.nanmax(dfx[stats]) > np.nanmax(binsx): 
        binsx.append(np.nanmax(dfx[stats]))
        stepx.append(binsx[-2]+bin_step)
    
    # Bin the stats data
    dfx['color'] = pd.cut(dfx[stats], bins=binsx, labels=stepx)
        
    # Group and count
    dfx = dfx.groupby(['lats','lons', 'color']).size().reset_index().rename(columns={0:'size'})
    
    return dfx, len(stepx)


# data presets
# dataset = 'mae'
# control = 'pr'
# bins    = [0,1,2,3,4,5]
# bin_step = 0.5
# lbls    = [0.5,1.5,2.5,3.5,4.5]
# mx_val  = 125
# wx        = 1
# leg     = ["<1","1$-$2","2$-$3","3$-$4","4$-$5",">5"]
# l_title = 'MAE (mm)'

dataset = 'me'
control = 'pr'
bins    = [-3,-2,-1,0,1,2,3]#[-2, -1, 0, 1, 2]
bin_step = 0.5
lbls    = [-2.5,-1.5,-0.5,0.5,1.5,2.5]#[-1.5, -0.5, 0.5, 1.5]
mx_val  = 125
wx        = 1
leg     = ["$\leq$-3", "(-3, -2]", "(-2, -1]","(-1, 0]","(0, 1]","(1, 2]","(2, 3]",">3"] #["$\leq$-2", "(-2, -1]","(-1, 0]","(0, 1]","(1, 2]",">2"]
l_title = 'Mean Precipitation Error (mm)'

# dataset = 'mre'
# control = 'pr'
# bins    = [0,50,100,150,200,250]
# bin_step = 25
# lbls    = [25,75,125,175,225]
# mx_val  = 175
# wx      = 50
# leg     = ["0$-$50","50$-$100","100$-$150","150$-$200","200$-$250",">250"]
# l_title = 'MRE (%)'

var     = ['opg', 'train', 'test']
titlex  = ['a) Actual OPG', 'b) Training OPG', 'c) Testing OPG']

# map presets
bounds  = [40.5, 49.5, -116, -105]
bbounds = [bounds[0]-0.75, bounds[1]+0.75, bounds[2]-0.75, bounds[3]+0.75]
datacrs = ccrs.PlateCarree()
cmap    = ncm.cmap('MPL_gist_gray')
ccmap   = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[0,32,44,81,111,146,168,199]) #ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[32,44,81,111,146,168])
tt_sz   = 24
lbl_sz  = 16


for cn in range(3):
    # pull variable
    dfx, step = form_bar_vals_3x4(df[['lons', 'lats', f'{dataset}_{control}_{var[cn]}']], 
                               f'{dataset}_{control}_{var[cn]}', bins.copy(), lbls.copy(), bin_step)

    # make plot
    fig = plt.figure(figsize=(5,6))
    ax = fig.add_axes( [0, 0, 1, 1], projection = ccrs.Mercator(
                                            central_longitude=np.mean(lons)))
    
    # add terrain
    ax.contourf(lons[0:359,263:672], lats[0:359,263:672], 
                elev[0:359,263:672], cmap=cmap, transform=datacrs, 
                levels=range(-500,4500,100), extend='upper', zorder=1)
    
    # add bar chart
    count = 0
    for idx in range(0, dfx.shape[0], step):
        lat = dfx.iloc[idx, 0]
        lon = dfx.iloc[idx, 1]
        
        if np.sum(dfx.iloc[idx:idx+step, 3]) > 0:
            ax_bar = fig.add_axes([(lon-(bbounds[2]))/((bbounds[3])-(bbounds[2]))-0.15 , 
                                   (lat-(bbounds[0]))/((bbounds[1])-(bbounds[0]))-0.11 , 
                                   0.3, 0.22])
            pcm = ax_bar.bar(dfx.iloc[idx:idx+step, 2], dfx.iloc[idx:idx+step, 3], 
                             color=ccmap.colors, width=wx, edgecolor=[0,0,0])
            
            ax_bar.set_ylim([0, mx_val])
            
            ax_bar.patch.set_alpha(0.01) # set background to transparent
            ax_bar.set_xticks([]) # remove tick labels
            ax_bar.set_yticks([0,50,100])
            ax_bar.set_yticklabels([]) # remove tick labels
            ax_bar.spines['top'].set_visible(False) # remove top line
            ax_bar.spines['right'].set_visible(False) # remove right line
            # ax_bar.spines['left'].set_visible(False) # remove left line
            #ax_bar.set_axis_off()
            
            if count == 0:
                ax_bar.set_yticklabels(['0','50','100'], rotation=90, 
                                       fontsize=lbl_sz, va='center')
                ax_bar.tick_params(axis='y', which='major', pad=10)
            
            count += 1
            
    ##### ADD LINES TO PLOT TO FIND WHAT YOU ARE EVEN DOING
    # ax.scatter(dfx.iloc[:, 1], dfx.iloc[:, 0], color='red', transform=datacrs)
    #####

    # Cartography
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", linewidth=2, zorder=2)
    ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", linewidth=2, zorder=3)
    ax.set_extent([bbounds[2], bbounds[3], bbounds[0], bbounds[1]])

    if cn == 0:
        legg = ax.legend(pcm, leg, fontsize=lbl_sz, ncols=6, 
                         bbox_to_anchor=(1, 0))
        legg.set_title(l_title, prop={'size':lbl_sz})
    
    # set title
    ax.set_title(titlex[cn], fontsize=tt_sz)

    # # Save and Show Figure
    # path = model_dir + model_name + "/"
    # plt.savefig(f"{path}map_{dataset}_{control}_vs_{var[cn]}2.png", dpi=300, 
    #             transparent=True, bbox_inches='tight')
    
    plt.show()
    plt.clf()




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
# add terrain
# ax[0].contourf(lons[0:359:5,263:672:5], lats[0:359:5,263:672:5], 
#             elev[0:359:5,263:672:5], cmap=cmap, transform=datacrs, 
#             levels=range(-500,4500,100), extend='upper', zorder=1)

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
# add terrain
# ax[1].contourf(lons[0:359:5,263:672:5], lats[0:359:5,263:672:5], 
#             elev[0:359:5,263:672:5], cmap=cmap, transform=datacrs, 
#             levels=range(-500,4500,100), extend='upper', zorder=1)

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
# add terrain
# ax[2].contourf(lons[0:359:5,263:672:5], lats[0:359:5,263:672:5], 
#             elev[0:359:5,263:672:5], cmap=cmap, transform=datacrs, 
#             levels=range(-500,4500,100), extend='upper', zorder=1)

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
# # cb_opg.set_ticklabels(['-10', '-8', '-6', '-4', '-2', '0'])
# cb_opg.set_ticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
cb_opg.set_label('Mean Precipitation Error (mm)', size=lbl_sz)
cb_opg.ax.tick_params(labelsize=lbl_sz)

plt.subplots_adjust(wspace=0.05, hspace=0.05)




# Save and Show Figure
path = model_dir + model_name + "/"
plt.savefig(f"{path}map_{dataset}_{control}_vs_all_heatmap2.png", dpi=300, 
            transparent=True, bbox_inches='tight')

plt.show()
plt.clf()




#%%

df = df.dropna()

#%% overlapped scatter plot


# def formulate_overlapped_scatter(dfx, roundType, roundSize, stats, bins, step):
#     # Remove rows with nan of stats
#     dfx = dfx.dropna(subset=stats)
    
#     # Round the lat and lons
#     if roundType == 'decimals':
#         dfx['lats'] = np.round(dfx['lats'], decimals=roundSize)
#         dfx['lons'] = np.round(dfx['lons'], decimals=roundSize)
#     elif roundType == 'integers':
#         dfx['lats'] = np.round(dfx['lats']/roundSize) * roundSize
#         dfx['lons'] = np.round(dfx['lons']/roundSize) * roundSize
    
#     # Create Bins
#     if np.nanmin(dfx[stats]) < np.nanmin(bins): 
#         bins.insert(0, np.nanmin(dfx[stats]))
#         step.insert(0, np.mean(bins[0:2]))
#     if np.nanmax(dfx[stats]) > np.nanmax(bins): 
#         bins.append(np.nanmax(dfx[stats]))
#         step.append(np.mean(bins[-2:]))

    
#     # Bin the stats data
#     dfx['color'] = pd.cut(dfx[stats], bins=bins, labels=step)
        
#     # Group and count
#     dfx = dfx.groupby(['lats','lons', 'color']).size().reset_index().rename(columns={0:'size'})
    
#     # remove zero counts
#     dfx = dfx[dfx['size'] != 0]
    
#     # Sort by largest count
#     dfx = dfx.sort_values(by=['size'], ascending=False)
    
#     return dfx


# df2 = formulate_overlapped_scatter(df[['lons', 'lats', 'mae_pr_test']], 'integers', 1.25, 
#                                    'mae_pr_test', [0,2,4,6,8,10,12], [1,3,5,7,9,11])

# # load regional location
# bounds = [40.5, 49.5, -116, -105]

# datacrs = ccrs.PlateCarree()
# cmap = ncm.cmap('MPL_gist_gray')
# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[28,44,81,111,146,176])


# fig = plt.figure(figsize=(5,6))
# ax = fig.add_axes( [0.1, 0.1, 0.8, 0.8], projection = ccrs.Mercator(central_longitude=np.mean(lons)))

# pcm = ax.contourf(lons[0:359,263:672], lats[0:359,263:672], elev[0:359,263:672], 
#                      cmap=cmap, transform=datacrs, levels=range(0,4500,100), 
#                      extend='upper', zorder=1)


# pcm = ax.scatter(df2['lons'],df2['lats'],s=df2['size']*15,c=df2['color'], 
#                  cmap=ccmap, transform=datacrs, zorder=4)
# pcm.set_clim(0,12)

# # Cartography
# ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black", zorder=2)
# ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black", zorder=3)
# ax.set_extent([bounds[2]-1, bounds[3]+1, bounds[0]-1.5, bounds[1]+1])

# # size legend
# cbar_opg = fig.add_axes([0.95, 0.5, 0.04, 0.4]) #[x0, y0, width, height]
# cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[0,2,4,6,8,10,12],#ticks=[-10, -8, -6, -4, -2, 0],
#                       extend='both', orientation='vertical', pad=0.0, 
#                       aspect=5, fraction=0.032)

# # legend1 = ax.legend(*pcm.legend_elements(), ['0-2 mm', '2-4 mm', '4-6 mm', '6-8 mm', '10-12 mm'],
# #                     bbox_to_anchor = (1.3, 1.0), title="MAE", fontsize=12)
# # ax.add_artist(legend1)

# handles, labels = pcm.legend_elements(prop="sizes", alpha=0.6, num=4)
# legend2 = ax.legend(handles, labels, bbox_to_anchor = (1.3, 0.5), 
#                     title="Num of Stations", fontsize=12, labelspacing=2)








# # plt.legend(*pcm.legend_elements("sizes", num=3))


# # Add colorbar
# # cbar_opg = fig.add_axes([0.16, 0.05, 0.7, 0.04]) #[x0, y0, width, height]
# # cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[0,2,4,6,8,10,12],#ticks=[-10, -8, -6, -4, -2, 0],
# #                       extend='both', orientation='horizontal', pad=0.0, 
# #                       aspect=5, fraction=0.032)
# # # cb_opg.set_ticklabels(['-10', '-8', '-6', '-4', '-2', '0'])
# # # cb_opg.set_ticklabels(['-11', '-9', '-7', '-5', '-3', '-1', '1'])
# # cb_opg.set_label('Mean Precipitation Error (mm)', size=16)
# # cb_opg.ax.tick_params(labelsize=16)

# plt.show()








#%%

print('Plotting Maps...')

cmap = ncm.cmap('MPL_gist_gray')
ccmap = ncm.cmapDiscrete('NCV_jaisnd', indexList=[255,229,203,177,151, 80,50,35,20,0])

# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=np.arange(160,0,-26))


ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[32,44,81,111,146,168])
#ccmap = ncm.cmapDiscrete('NCV_jet', np.arange(255,0,-22))

# ccmap = ncm.cmapDiscrete('BlueDarkRed18', indexList=[0,1,2,3,4,5,6, 11,12,13,14,15,16,17])
# ccmap = ncm.cmapDiscrete('GreenMagenta16', indexList=[0,1,2,3,4,5, 10,11,12,13,14,15])
# ccmap = ncm.cmapDiscrete('cmp_flux', indexList=[21,20,19,18,17,16,15,14,13,12, 9,8,7,6,5,4,3,2,1,0])

cclim = 13
size = 9
aspect = 0.01
# 
# load regional location
bounds = [40.5, 49.5, -116, -105]

# Create Figure
datacrs = ccrs.PlateCarree()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 6),
                        subplot_kw={'projection': ccrs.Mercator(central_longitude=np.mean(lons))})

#### ACTUAL
# Shade terrain
pcm = ax[0].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['mre_pr_opg'], ascending=False)
pcm = ax[0].scatter(df['lons'], df['lats'], s=size, c=df['mre_pr_opg'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(0,300)

# Cartography
ax[0].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[0].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[0].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[0].set_title("a) Actual OPG", fontsize=16)

#### TRAINING
# Shade terrain
pcm = ax[1].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['mre_pr_train'], ascending=False)
pcm = ax[1].scatter(df['lons'], df['lats'], s=size, c=df['mre_pr_train'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(0,300)

# Cartography
ax[1].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[1].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[1].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[1].set_title("b) Trained OPG", fontsize=16)

#### TESTING
# Shade terrain
pcm = ax[2].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['mre_pr_test'], ascending=False)
pcm = ax[2].scatter(df['lons'], df['lats'], s=size, c=df['mre_pr_test'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(0,300)

# Cartography
ax[2].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[2].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[2].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[2].set_title("c) Tested OPG", fontsize=16)

# Add colorbar
cbar_opg = fig.add_axes([0.16, 0.16, 0.7, 0.04]) #[x0, y0, width, height]
cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[0,50,100,150,200,250],#ticks=[-10, -8, -6, -4, -2, 0],
                      extend='max', orientation='horizontal', pad=0.0, 
                      aspect=5, fraction=0.032)
# cb_opg.set_ticklabels(['-10', '-8', '-6', '-4', '-2', '0'])
cb_opg.set_ticklabels(['0', '50', '100', '150', '200', '250'])
cb_opg.set_label('Mean Relative Precipitation Error (%)', size=16)
cb_opg.ax.tick_params(labelsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.05)


# Save and Show Figure
path = model_dir + model_name + "/"
# plt.savefig(f"{path}mean_relative_precip_error_map.png", dpi=200, 
#             transparent=True, bbox_inches='tight')

plt.show()

#%% Plot try pcolormesh

print('Plotting Maps...')

cmap = ncm.cmap('MPL_gist_gray')
ccmap = ncm.cmapDiscrete('NCV_jaisnd', indexList=[255,229,203,177,151, 80,50,35,20,0])

# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=np.arange(160,0,-26))


ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[32,44,81,111,146,168])
#ccmap = ncm.cmapDiscrete('NCV_jet', np.arange(255,0,-22))

# ccmap = ncm.cmapDiscrete('BlueDarkRed18', indexList=[0,1,2,3,4,5,6, 11,12,13,14,15,16,17])
# ccmap = ncm.cmapDiscrete('GreenMagenta16', indexList=[0,1,2,3,4,5, 10,11,12,13,14,15])
# ccmap = ncm.cmapDiscrete('cmp_flux', indexList=[21,20,19,18,17,16,15,14,13,12, 9,8,7,6,5,4,3,2,1,0])

cclim = 13
size = 9
aspect = 0.01
# 
# load regional location
bounds = [40.5, 49.5, -116, -105]

# Create Figure
datacrs = ccrs.PlateCarree()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 6),
                        subplot_kw={'projection': ccrs.Mercator(central_longitude=np.mean(lons))})

#### ACTUAL
# Shade terrain
pcm = ax[0].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['mae_pr_opg'], ascending=False)
pcm = ax[0].scatter(df['lons'], df['lats'], s=size, c=df['mae_pr_opg'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(0,6)

# Cartography
ax[0].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[0].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[0].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[0].set_title("a) Actual OPG", fontsize=16)

#### TRAINING
# Shade terrain
pcm = ax[1].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['mae_pr_train'], ascending=False)
pcm = ax[1].scatter(df['lons'], df['lats'], s=size, c=df['mae_pr_train'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(0,6)

# Cartography
ax[1].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[1].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[1].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[1].set_title("b) Trained OPG", fontsize=16)

#### TESTING
# Shade terrain
pcm = ax[2].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['mae_pr_test'], ascending=False)
pcm = ax[2].scatter(df['lons'], df['lats'], s=size, c=df['mae_pr_test'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(0,6)

# Cartography
ax[2].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[2].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[2].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[2].set_title("c) Tested OPG", fontsize=16)

# Add colorbar
cbar_opg = fig.add_axes([0.16, 0.16, 0.7, 0.04]) #[x0, y0, width, height]
cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[0,1,2,3,4,5],#ticks=[-10, -8, -6, -4, -2, 0],
                      extend='max', orientation='horizontal', pad=0.0, 
                      aspect=5, fraction=0.032)
# cb_opg.set_ticklabels(['-10', '-8', '-6', '-4', '-2', '0'])
cb_opg.set_ticklabels(['0', '1', '2', '3', '4', '5'])
cb_opg.set_label('Mean Absolute Precipitation Error (mm)', size=16)
cb_opg.ax.tick_params(labelsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.05)


# Save and Show Figure
path = model_dir + model_name + "/"
# plt.savefig(f"{path}mean_absolute_precip_error_map.png", dpi=200, 
#             transparent=True, bbox_inches='tight')

plt.show()


#%% Plot as Scatter Plot

print('Plotting Maps...')

cmap = ncm.cmap('MPL_gist_gray')
ccmap = ncm.cmapDiscrete('NCV_jaisnd', indexList=[255,229,203,177,151, 80,50,35,20,0])

# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=np.arange(160,0,-26))


ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[32,44,81,111,146,168])
#ccmap = ncm.cmapDiscrete('NCV_jet', np.arange(255,0,-22))

# ccmap = ncm.cmapDiscrete('BlueDarkRed18', indexList=[0,1,2,3,4,5,6, 11,12,13,14,15,16,17])
# ccmap = ncm.cmapDiscrete('GreenMagenta16', indexList=[0,1,2,3,4,5, 10,11,12,13,14,15])
# ccmap = ncm.cmapDiscrete('cmp_flux', indexList=[21,20,19,18,17,16,15,14,13,12, 9,8,7,6,5,4,3,2,1,0])

cclim = 13
size = 9
aspect = 0.01
# 
# load regional location
bounds = [40.5, 49.5, -116, -105]

# Create Figure
datacrs = ccrs.PlateCarree()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 6),
                        subplot_kw={'projection': ccrs.Mercator(central_longitude=np.mean(lons))})

#### ACTUAL
# Shade terrain
pcm = ax[0].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['me_pr_opg'], key=abs)
pcm = ax[0].scatter(df['lons'], df['lats'], s=size, c=df['me_pr_opg'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(-3,3)

# Cartography
ax[0].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[0].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[0].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[0].set_title("a) Actual OPG", fontsize=16)

#### TRAINING
# Shade terrain
pcm = ax[1].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['me_pr_train'], key=abs)
pcm = ax[1].scatter(df['lons'], df['lats'], s=size, c=df['me_pr_train'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(-3,3)

# Cartography
ax[1].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[1].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[1].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[1].set_title("b) Trained OPG", fontsize=16)

#### TESTING
# Shade terrain
pcm = ax[2].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# Add OPG Error Scatter
df = df.sort_values(by=['me_pr_test'], key=abs)
pcm = ax[2].scatter(df['lons'], df['lats'], s=size, c=df['me_pr_test'], 
                    cmap=ccmap, transform=datacrs)
# pcm.set_clim(cclim*-1,2)
pcm.set_clim(-3,3)

# Cartography
ax[2].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
ax[2].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
ax[2].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# Add Title
ax[2].set_title("c) Tested OPG", fontsize=16)

# Add colorbar
cbar_opg = fig.add_axes([0.16, 0.16, 0.7, 0.04]) #[x0, y0, width, height]
cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[-3,-2,-1,0,1,2,3],#ticks=[-10, -8, -6, -4, -2, 0],
                      extend='both', orientation='horizontal', pad=0.0, 
                      aspect=5, fraction=0.032)
# cb_opg.set_ticklabels(['-10', '-8', '-6', '-4', '-2', '0'])
cb_opg.set_ticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
cb_opg.set_label('Mean Precipitation Error (mm)', size=16)
cb_opg.ax.tick_params(labelsize=16)

plt.subplots_adjust(wspace=0.05, hspace=0.05)


# Save and Show Figure
path = model_dir + model_name + "/"
plt.savefig(f"{path}mean_precip_error_map.png", dpi=200, 
            transparent=True, bbox_inches='tight')

plt.show()

#%% Make histograms separately

# print('Plotting Histograms...')

# bb = [-10000, -11, -9, -7, -5, -3, -1, 1, 10000]
# xx = [-12,-10,-8,-6,-4,-2,0,2]
# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=np.arange(160,0,-26))
# colors = [[100/255,20/255,146/255],[174/255,0/255,65/255],[255/255,20/255,0],[255/255,119/255,0/255],
#           [255/255,214/255,0/255],[67/255,197/255,11/255],[0,31/255,255/255],[0,194/255,255/255]]

# # Create Figure
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 2), sharey=True)

# ##### Actual
# counts, bins = np.histogram(mean_opg, bins=bb)
# ax[0].bar(xx, counts, width=2, align='center', color=colors, edgecolor='black')
# ax[0].set_xticks(bb[1:8], fontsize=12)
# ax[0].set_xlim((-13,3))
# ax[0].set_yticks([0,200,400,600,800], fontsize=14)
# ax[0].grid()
# ax[0].set_axisbelow(True)

# ##### Train
# counts, bins = np.histogram(mean_train, bins=bb)
# ax[1].bar(xx, counts, width=2, align='center', color=colors, edgecolor='black')
# ax[1].set_xticks(bb[1:8], fontsize=12)
# ax[1].set_xlim((-13,3))
# ax[1].set_yticks([0,200,400,600,800], fontsize=14)
# ax[1].grid()
# ax[1].set_axisbelow(True)

# ##### Test
# counts, bins = np.histogram(mean_test, bins=bb)
# ax[2].bar(xx, counts, width=2, align='center', color=colors, edgecolor='black')
# ax[2].set_xticks(bb[1:8], fontsize=12)
# ax[2].set_xlim((-13,3))
# ax[2].set_yticks([0,200,400,600,800], fontsize=14)
# ax[2].grid()
# ax[2].set_axisbelow(True)

# ax[2].set_ylim((0,800))
# ax[0].set_ylabel('Count of Stations', fontsize=12)
# plt.subplots_adjust(wspace=0.05, hspace=0.05)

# path = model_dir + model_name + "/"
# plt.savefig(f"{path}mean_precip_error_hist.png", dpi=200, 
#             transparent=True, bbox_inches='tight')


#%% Make histograms separately

# print('Plotting Scatter Plots...')

# bb = [-10000, -11, -9, -7, -5, -3, -1, 1, 10000]
# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[176,161,146,131,111,81,28,44])
# size = 10

# # Create Figure
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 2), sharey=True)

# ##### Actual
# ax[0].scatter(mean_opg, ghcnd_elev, s=size, c=mean_opg, cmap=ccmap)
# ax[0].set_xticks(bb[1:8], fontsize=12)
# ax[0].set_xlim((-25,5))
# # ax[0].set_yticks([0,250,500,750,1000], fontsize=14)
# ax[0].grid()
# ax[0].set_axisbelow(True)


# ##### Train
# ax[1].scatter(mean_train, ghcnd_elev, s=size, c=mean_train, cmap=ccmap)
# ax[1].set_xticks(bb[1:8], fontsize=12)
# ax[1].set_xlim((-25,5))
# # ax[1].set_yticks([0,250,500,750,1000], fontsize=14)
# ax[1].grid()
# ax[1].set_axisbelow(True)

# ##### Test
# ax[2].scatter(mean_test, ghcnd_elev, s=size, c=mean_test, cmap=ccmap)
# ax[2].set_xticks(bb[1:8], fontsize=12)
# ax[2].set_xlim((-25,5))
# # ax[2].set_yticks([0,250,500,750,1000], fontsize=14)
# ax[2].grid()
# ax[2].set_axisbelow(True)

# # ax[2].set_ylim((0,1000))
# ax[0].set_ylabel('Elevation', fontsize=13)
# plt.subplots_adjust(wspace=0.05, hspace=0.05)

# path = model_dir + model_name + "/"
# plt.savefig(f"{path}mean_precip_error_hist.png", dpi=200, 
#             transparent=True, bbox_inches='tight')





#%% Plot as Scatter Plot

# print('Plotting Maps...')

# cmap = ncm.cmap('MPL_gist_gray')
# ccmap = ncm.cmapDiscrete('NCV_jaisnd', indexList=[255,229,203,177,151, 80,50,35,20,0])

# # ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=np.arange(160,0,-26))


# ccmap = ncm.cmapDiscrete('BkBlAqGrYeOrReViWh200', indexList=[28,81,111,131,146,161,176])


# size = 9
# aspect = 0.01
# # 
# # load regional location
# bounds = [40.5, 49.5, -116, -105]

# # Create Figure
# datacrs = ccrs.PlateCarree()
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 6),
#                        subplot_kw={'projection': ccrs.Mercator(central_longitude=np.mean(lons))})

# #### ACTUAL
# # Shade terrain
# pcm = ax[0].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
#                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# # Add OPG Error Scatter
# idx = np.argsort(mean_re_opg)
# pcm = ax[0].scatter(ghcnd_latlon[idx,1], ghcnd_latlon[idx,0], s=size, 
#                     c=mean_re_opg[idx], cmap=ccmap, transform=datacrs)

# pcm.set_clim(0, 175)

# # Cartography
# ax[0].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
# ax[0].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
# ax[0].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
# ax[0].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# # Add Title
# ax[0].set_title("a) Actual OPG", fontsize=16)

# #### TRAINING
# # Shade terrain
# pcm = ax[1].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
#                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# # Add OPG Error Scatter
# idx = np.argsort(mean_re_train)
# pcm = ax[1].scatter(ghcnd_latlon[idx,1], ghcnd_latlon[idx,0], s=size, 
#                     c=mean_re_train[idx], cmap=ccmap, transform=datacrs)

# pcm.set_clim(0, 175)

# # Cartography
# ax[1].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
# ax[1].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
# ax[1].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
# ax[1].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# # Add Title
# ax[1].set_title("b) Trained OPG", fontsize=16)

# #### TESTING
# # Shade terrain
# pcm = ax[2].contourf(lons[47:311,311:622], lats[47:311,311:622], elev[47:311,311:622], 
#                      cmap=cmap, transform=datacrs, levels=range(200,4100,100), extend='max')

# # Add OPG Error Scatter
# idx = np.argsort(mean_re_test)
# pcm = ax[2].scatter(ghcnd_latlon[idx,1], ghcnd_latlon[idx,0], s=size, 
#                     c=mean_re_test[idx], cmap=ccmap, transform=datacrs)

# pcm.set_clim(0, 175)

# # Cartography
# ax[2].add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="black")
# ax[2].add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="black")
# ax[2].add_feature(cfeat.STATES.with_scale('50m'), edgecolor="black")
# ax[2].set_extent([bounds[2]-1, bounds[3]+0.5, bounds[0]-0.5, bounds[1]+0.5])

# # Add Title
# ax[2].set_title("c) Tested OPG", fontsize=16)

# # Add colorbar
# cbar_opg = fig.add_axes([0.16, 0.16, 0.7, 0.04]) #[x0, y0, width, height]
# cb_opg = fig.colorbar(pcm, cax=cbar_opg, ticks=[0, 50, 100, 150],
#                       extend='max', orientation='horizontal', pad=0.0, 
#                       aspect=5, fraction=0.032)
# # cb_opg.set_ticklabels(['-10', '-8', '-6', '-4', '-2', '0'])
# cb_opg.set_ticklabels(['0', '50', '100', '150'])
# cb_opg.set_label('Mean Relative Precipitation Error (%)', size=16)
# cb_opg.ax.tick_params(labelsize=16)

# plt.subplots_adjust(wspace=0.05, hspace=0.05)


# Save and Show Figure
# path = model_dir + model_name + "/"
# plt.savefig(f"{path}mean_precip_relative_error_map.png", dpi=200, 
#             transparent=True, bbox_inches='tight')

#%% Make histograms separately

# print('Plotting Histograms...')

# wx = 25
# bb = [0, 25, 50, 75, 100, 125, 150, 10000]
# xx = [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5]
# colors = [[0/255,31/255,254/255],[67/255,197/255,11/255],[255/255,214/255,0],[255/255,119/255,1/255],
#           [250/255,22/255,4/255],[174/255,0/255,64/255],[100/255,19/255,147/255]]

# # Create Figure
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 2), sharey=True)

# ##### Actual
# counts, bins = np.histogram(mean_re_opg, bins=bb)
# ax[0].bar(xx, counts, width=wx, align='center', color=colors, edgecolor='black')
# ax[0].set_xticks(bb[1:8], fontsize=12)
# ax[0].set_xlim((0,175))
# ax[0].set_yticks([0,200,400,600,800,1000], fontsize=14)
# ax[0].grid()
# ax[0].set_axisbelow(True)

# ##### Train
# counts, bins = np.histogram(mean_re_train, bins=bb)
# ax[1].bar(xx, counts, width=wx, align='center', color=colors, edgecolor='black')
# ax[1].set_xticks(bb[1:8], fontsize=12)
# ax[1].set_xlim((0,175))
# ax[1].set_yticks([0,200,400,600,800,1000], fontsize=14)
# ax[1].grid()
# ax[1].set_axisbelow(True)

# ##### Test
# counts, bins = np.histogram(mean_re_test, bins=bb)
# ax[2].bar(xx, counts, width=wx, align='center', color=colors, edgecolor='black')
# ax[2].set_xticks(bb[1:8], fontsize=12)
# ax[2].set_xlim((0,175))
# ax[2].set_yticks([0,200,400,600,800,1000], fontsize=14)
# ax[2].grid()
# ax[2].set_axisbelow(True)

# ax[2].set_ylim((0,1000))
# ax[0].set_ylabel('Count of Stations', fontsize=12)
# plt.subplots_adjust(wspace=0.05, hspace=0.05)

# path = model_dir + model_name + "/"
# plt.savefig(f"{path}mean_precip_relative_error_hist.png", dpi=200, 
#             transparent=True, bbox_inches='tight')
