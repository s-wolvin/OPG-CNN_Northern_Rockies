"""
Savanna Wolvin
Created: Apr 10th, 2024
Edited: Apr 30th, 2024


##### SUMMARY #################################################################
This python script is designed to formulate the errors in predicted 
precipitation to the observed precipitation. Predicted precipitation is 
formulated from the true OPG values, the predicted training OPG values, and the 
predicted testing OPG values. The observed precipitation is the Global 
Historical Climatological Network (GHCN) - Daily data. The ERA5 at 0.5 degree 
is also plotted, after interpreting to the observation location and adjusting 
the precipitation value based on quantile mapping. 

This script is edited from a script from Michael Pletcher - michael.pletcher@utah.edu, 
originally described in this publication:

    Roebber, P. J., 2009: Visualizing Multiple Measures of Forecast Quality. 
    Wea. Forecasting, 24, 601608, https://doi.org/10.1175/2008WAF2222159.1.
    

##### INPUT ###################################################################
fi_dir      - Directory to the facet data
opg_dir     - Directory to the OPG data
model_dir   - Directory to the CNN model data
model_name  - File name of the desired CNN model
d_dir       - Directory to the ERA5 data
years       - Years desired
months      - Months desired


"""
#%% Global Imports

import scipy.io as sio
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
import sys
sys.path.insert(1, '/uufs/chpc.utah.edu/common/home/u1324060/nclcmappy/')
import warnings


warnings.filterwarnings('ignore')

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

mat_file = sio.loadmat(fi_dir + 'station_count')  
rm_opg   = mat_file['num'] < 3


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
era5 = xr.Dataset()
for data_yearsX in tqdm(np.arange(years[0], years[1]+1)): # loop through each year 
    # Access the NC File and Convert Dataset to an Xarray
    ncfile = xr.open_dataset(
        d_dir + "daily/sfc/era5_precip_" + str(data_yearsX) + "_oct-apr_daily.nc")
    
    # Save Atmospheric Variable
    era5 = xr.merge([era5, ncfile])



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
        # same elevation average
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



#%% remove unused stations

print("Remove unused stations...")

# find the stations that lack observations for both the training and testing subsets
idx = np.sum(~np.isnan(cnn_pr_train), axis=1) + np.sum(~np.isnan(cnn_pr_test), axis=1)
idx[idx > 0] = 1

# Pull station data of observed stations
ghcnd_pr_       = ghcnd_pr[:, idx==1]
ghcnd_elev_     = ghcnd_elev[idx==1]
ghcnd_latlon_   = ghcnd_latlon[idx==1,:]
ghcnd_assgnmnt_ = ghcnd_assgnmnt[idx==1,:]

# Pull precip data of observed stations
opg_pr_         = opg_pr[idx==1, :]
cnn_pr_train_   = cnn_pr_train[idx==1, :]
cnn_pr_test_    = cnn_pr_test[idx==1, :]



#%% remove spring, summer, fall

print("Remove spring, summer, and fall...")

# determine index of winter dates
dts = pd.DataFrame({'date': ghcnd_days})
dts['date'] = pd.to_datetime(dts['date'], format='%Y-%m-%d')
dates_idx= dts['date'].dt.month.isin([12,1,2])

# subset dates, days of observations, and observed OPGs
ghcnd_days = ghcnd_days[dates_idx]
ghcnd_pr_ = ghcnd_pr_[dates_idx,:]
opg_pr_ = opg_pr_[:, dates_idx]

# remove one weird value below zero
ghcnd_pr_[ghcnd_pr_ < 0] = 0



#%% Formulate daily precip at each station from the ERA5

print("Linearly interpolate ERA5 precip...")

# create array
era5_pr = np.zeros((np.shape(ghcnd_days)[0], np.shape(ghcnd_latlon_)[0]))

# linearly interpolate
for stx in tqdm(range(np.shape(ghcnd_latlon_)[0])):
    xx = era5.interp(latitude=ghcnd_latlon_[stx,0],
                     longitude=ghcnd_latlon_[stx,1])
    xx = xx.isel(time=xx.time.isin(ghcnd_days.astype('datetime64')))
    era5_pr[:, stx] = xx['precip'].values


# convert meters to milimeters
era5_pr = era5_pr * 1000



#%% Quantile Mapping of the ERA5 precipitation to the observations

print("Quantile mapping...")

# new empty array
era5_pr_ = np.zeros(np.shape(era5_pr))

# Loop through each station
for stx in tqdm(range(np.shape(ghcnd_latlon_)[0])):
    era5_cdf = ECDF(era5_pr[era5_pr[:, stx] != 0, stx]) # create CDF function
    
    # loop through each day
    for dayx in range(np.shape(era5_pr)[0]):
        if era5_pr[dayx, stx] != 0: # if there is an era5 obs
            quant = era5_cdf(era5_pr[dayx, stx]) # determine quantile
            value = np.nanquantile(ghcnd_pr_[ghcnd_pr_[:,stx] != 0.0, stx], quant) # pull value at quantile of obs
            era5_pr_[dayx, stx] = value # save value



#%% set opg that are zero to nan

print("Process zeros and NaNs...")

# Values below zero or a threshold are set to zero
opg_pr_[opg_pr_ < 0] = 0
cnn_pr_test_[cnn_pr_test_ < 0] = 0
cnn_pr_train_[cnn_pr_train_ < 0] = 0
era5_pr_[era5_pr_ < 0.001] = 0

# If the value is zero, set to NaN
# We don't know if it was because of no-precipitation or no-observations
opg_pr_[opg_pr_ == 0] = np.nan
cnn_pr_test_[cnn_pr_test_ == 0] = np.nan
cnn_pr_train_[cnn_pr_train_ == 0] = np.nan
era5_pr_[era5_pr_ == 0] = np.nan



#%%

print("Create index array of training/testing days...")

# pull the times for testing and training
ghcnd_train = np.zeros(np.shape(ghcnd_days))
for dx in train_time:
    ghcnd_train = ghcnd_train + np.where(dx == ghcnd_days.astype('datetime64'), 1, 0)
    
ghcnd_test = np.zeros(np.shape(ghcnd_days))
for dx in test_time:
    ghcnd_test = ghcnd_test + np.where(dx == ghcnd_days.astype('datetime64'), 1, 0)
    
    

#%%
"""
if there is not an OPG recorded, and the GHCND is positive, THEN there weren't enough stations, SO the precip value is NaN
if there is     an OPG recorded, and the GHCND is positive, THEN the precip value was negative, SO the precip value is zero
 """

print("Should the predicted value be NaN or set to zero...")


for assgn in range(len(ghcnd_assgnmnt_)):
    
    # pull the OPGs for the associated station
    st_opg = act_opg[dates_idx==1, ghcnd_assgnmnt_[assgn][0].astype('int')-1]
    
    # process act opg
    where = np.where(((np.isnan(st_opg.T) & (ghcnd_pr_[:, assgn] > 0))),1,0)
    opg_pr_[assgn, where] = 0

    # process training opgs
    where = np.where(((np.isnan(st_opg[ghcnd_train==1]) & (ghcnd_pr_[ghcnd_train==1, assgn] > 0))),1,0)
    cnn_pr_train_[assgn, where] = 0

    # process testing opgs
    where = np.where(((np.isnan(st_opg[ghcnd_test==1]) & (ghcnd_pr_[ghcnd_test==1, assgn] > 0))),1,0)
    cnn_pr_test_[assgn, where] = 0
    
    # process era5
    where = np.where(((np.isnan(st_opg.T) & (ghcnd_pr_[:, assgn] > 0))),1,0)
    era5_pr_[where, assgn] = 0
    



#%% Functions for model statistics
# Hit rate
def hit_rate(a, c):
    return a / (a + c)

# False alarm ratio (FAR)
def false_alarm_ratio(a, b):
    return b / (a + b)

# Critical success index (CSI)
def critical_success_index(a, b, c):
    return a / (a + b + c)


#%% 
""" Formulate validation statistics
"""

percentiles = ['5','25','50','75','95']

# Create DataFrames for statistics
act_df = pd.DataFrame()
act_df['percentiles'] = percentiles

tra_df = pd.DataFrame()
tra_df['percentiles'] = percentiles

tes_df = pd.DataFrame()
tes_df['percentiles'] = percentiles

era_df = pd.DataFrame()
era_df['percentiles'] = percentiles


# Calculate validation statistics relative to observed snowfall percentiles
for percentile in percentiles:
    print(percentile)
    for stx in tqdm(range(len(ghcnd_elev_))):
        
######### Actual OPG vs GHCND #######################################

        obs_col   = ghcnd_pr_[:, stx]
        obs_col[obs_col == 0] = np.nan
        model_col = opg_pr_[stx, :]
        
        # Calculate thresholds
        fcst_threshold = np.nanpercentile(model_col, np.array(percentile, dtype = float))
        obs_threshold  = np.nanpercentile(obs_col, np.array(percentile, dtype = float))

        # Identify hits
        hits = np.where(((model_col >= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        hits = np.sum(hits == 1)

        # Identify false alarms
        false_alarm = np.where(((model_col >= obs_threshold) & (obs_col <= obs_threshold)), 1, np.nan)
        false_alarm = np.sum(false_alarm == 1)

        # Identify misses
        misses = np.where(((model_col <= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        misses = np.sum(misses == 1)
        
        # Calculate validation metrics
        hit_rates    = hit_rate(hits, misses)
        false_alarms = false_alarm_ratio(hits, false_alarm)
        csis         = critical_success_index(hits, false_alarm, misses)
        sr           = 1 - false_alarms
        
        # Calculate a few statistics based on percentiles
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_#_pr_obs_exceeding_prct']   = np.sum(obs_col   >= obs_threshold)
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_#_pr_fcsts_exceeding_prct'] = np.sum(model_col >= obs_threshold)
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_hit_rate_prct']             = hit_rates
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_fa_ratio']                  = false_alarms
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_critial_success_idx']       = csis
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_success_ratio']             = sr

        # Calculate event sizes based on percentiles for observed and forecasted snowfall events
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + '_pr_event_size_by_prct'] = fcst_threshold
        act_df.loc[act_df['percentiles'] == percentile, str(stx+1) + 'obs_pr_event_size_by_prct'] = obs_threshold
        
        
######### Train OPG VS GHCND #####################################

        obs_col   = ghcnd_pr_[ghcnd_train==1, stx]
        obs_col[obs_col == 0] = np.nan
        model_col = cnn_pr_train_[stx, :]
        
        # Calculate thresholds
        fcst_threshold = np.nanpercentile(model_col, np.array(percentile, dtype = float))
        obs_threshold  = np.nanpercentile(obs_col, np.array(percentile, dtype = float))

        # Identify hits
        hits = np.where(((model_col >= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        hits = np.sum(hits == 1)

        # Identify false alarms
        false_alarm = np.where(((model_col >= obs_threshold) & (obs_col <= obs_threshold)), 1, np.nan)
        false_alarm = np.sum(false_alarm == 1)

        # Identify misses
        misses = np.where(((model_col <= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        misses = np.sum(misses == 1)
        
        # Calculate validation metrics
        hit_rates    = hit_rate(hits, misses)
        false_alarms = false_alarm_ratio(hits, false_alarm)
        csis         = critical_success_index(hits, false_alarm, misses)
        sr           = 1 - false_alarms
        
        # Calculate a few statistics based on percentiles
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_#_pr_obs_exceeding_prct']   = np.sum(obs_col   >= obs_threshold)
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_#_pr_fcsts_exceeding_prct'] = np.sum(model_col >= obs_threshold)
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_hit_rate_prct']             = hit_rates
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_fa_ratio']                  = false_alarms
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_critial_success_idx']       = csis
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_success_ratio']             = sr

        # Calculate event sizes based on percentiles for observed and forecasted snowfall events
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + '_pr_event_size_by_prct'] = fcst_threshold
        tra_df.loc[tra_df['percentiles'] == percentile, str(stx+1) + 'obs_pr_event_size_by_prct'] = obs_threshold      


######### Test OPG VS GHCND ####################################
        
        obs_col   = ghcnd_pr_[ghcnd_test==1, stx]
        obs_col[obs_col == 0] = np.nan
        model_col = cnn_pr_test_[stx, :]
        
        # Calculate thresholds
        fcst_threshold = np.nanpercentile(model_col, np.array(percentile, dtype = float))
        obs_threshold  = np.nanpercentile(obs_col, np.array(percentile, dtype = float))

        # Identify hits
        hits = np.where(((model_col >= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        hits = np.sum(hits == 1)

        # Identify false alarms
        false_alarm = np.where(((model_col >= obs_threshold) & (obs_col <= obs_threshold)), 1, np.nan)
        false_alarm = np.sum(false_alarm == 1)

        # Identify misses
        misses = np.where(((model_col <= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        misses = np.sum(misses == 1)
        
        # Calculate validation metrics
        hit_rates    = hit_rate(hits, misses)
        false_alarms = false_alarm_ratio(hits, false_alarm)
        csis         = critical_success_index(hits, false_alarm, misses)
        sr           = 1 - false_alarms
        
        # Calculate a few statistics based on percentiles
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_#_pr_obs_exceeding_prct']   = np.sum(obs_col   >= obs_threshold)
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_#_pr_fcsts_exceeding_prct'] = np.sum(model_col >= obs_threshold)
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_hit_rate_prct']             = hit_rates
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_fa_ratio']                  = false_alarms
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_critial_success_idx']       = csis
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_success_ratio']             = sr

        # Calculate event sizes based on percentiles for observed and forecasted snowfall events
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_pr_event_size_by_prct'] = fcst_threshold
        tes_df.loc[tes_df['percentiles'] == percentile, str(stx+1) + '_obs_pr_event_size_by_prct'] = obs_threshold        
           
        
######### QM ERA5 VS GHCND ####################################
        
        obs_col   = ghcnd_pr_[:, stx]
        obs_col[obs_col == 0] = np.nan
        model_col = era5_pr_[:, stx]
        
        # Calculate thresholds
        fcst_threshold = np.nanpercentile(model_col, np.array(percentile, dtype = float))
        obs_threshold  = np.nanpercentile(obs_col, np.array(percentile, dtype = float))

        # Identify hits
        hits = np.where(((model_col >= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        hits = np.sum(hits == 1)

        # Identify false alarms
        false_alarm = np.where(((model_col >= obs_threshold) & (obs_col <= obs_threshold)), 1, np.nan)
        false_alarm = np.sum(false_alarm == 1)

        # Identify misses
        misses = np.where(((model_col <= obs_threshold) & (obs_col >= obs_threshold)), 1, np.nan)
        misses = np.sum(misses == 1)
        
        # Calculate validation metrics
        hit_rates    = hit_rate(hits, misses)
        false_alarms = false_alarm_ratio(hits, false_alarm)
        csis         = critical_success_index(hits, false_alarm, misses)
        sr           = 1 - false_alarms
        
        # Calculate a few statistics based on percentiles
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_#_pr_obs_exceeding_prct']   = np.sum(obs_col   >= obs_threshold)
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_#_pr_fcsts_exceeding_prct'] = np.sum(model_col >= obs_threshold)
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_hit_rate_prct']             = hit_rates
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_fa_ratio']                  = false_alarms
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_critial_success_idx']       = csis
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_success_ratio']             = sr

        # Calculate event sizes based on percentiles for observed and forecasted snowfall events
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_pr_event_size_by_prct'] = fcst_threshold
        era_df.loc[era_df['percentiles'] == percentile, str(stx+1) + '_obs_pr_event_size_by_prct'] = obs_threshold  
        
        
        
#%% Save data

act_df.to_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/actual_success.csv')
tra_df.to_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/training_success.csv')
tes_df.to_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/testing_success.csv')
era_df.to_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/era5_success.csv')


#%% load in data

act_df = pd.read_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/actual_success.csv')
tra_df = pd.read_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/training_success.csv')
tes_df = pd.read_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/testing_success.csv')
era_df = pd.read_csv('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/contour_success_index/era5_success.csv')


#%% formulate metrics for each possibility

act_hit_rates     = act_df.filter(like='_hit_rate_prct', axis=1)
act_success_ratio = act_df.filter(like='_success_ratio', axis=1)
act_hit_rates     = act_hit_rates.mean(axis=1)
act_success_ratio = act_success_ratio.mean(axis=1)

tra_hit_rates     = tra_df.filter(like='_hit_rate_prct', axis=1)
tra_success_ratio = tra_df.filter(like='_success_ratio', axis=1)
tra_hit_rates     = tra_hit_rates.mean(axis=1)
tra_success_ratio = tra_success_ratio.mean(axis=1)

tes_hit_rates     = tes_df.filter(like='_hit_rate_prct', axis=1)
tes_success_ratio = tes_df.filter(like='_success_ratio', axis=1)
tes_hit_rates     = tes_hit_rates.mean(axis=1)
tes_success_ratio = tes_success_ratio.mean(axis=1)

era_hit_rates     = era_df.filter(like='_hit_rate_prct', axis=1)
era_success_ratio = era_df.filter(like='_success_ratio', axis=1)
era_hit_rates     = era_hit_rates.mean(axis=1)
era_success_ratio = era_success_ratio.mean(axis=1)



#%% Plot

fig, ax = plt.subplots(1, 1, figsize=(9.5, 8), facecolor='w')
grid_ticks  = np.arange(0, 1.01, 0.01)
sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
bias        = pod_g / sr_g
csi         = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)

axis_label_sz = 20
tick_label_sz = 16
s = 5

# Plot contours for CSI
csi_contour = plt.contourf(sr_g, pod_g, csi, np.arange(0, 1.01, 0.1), 
                           cmap= "Greys", alpha = 0.825)
csi_contourr = plt.contour(sr_g, pod_g, csi, np.arange(0, 1.01, 0.1), 
                           colors = 'k', linewidths = 0.35)
# for xi in np.arange(0.1, 1, 0.1):
#     plt.text(1.01, xi-0.01, str(np.round(xi, decimals=1)), fontsize=tick_label_sz)

# Plot Contours for BIAS
b_contour   = plt.contour(sr_g, pod_g, bias, [0.25, 0.5, 1, 2, 4], colors="k", 
                          linestyles="dashed")
clabels = plt.clabel(b_contour, fmt="%1.1f", manual=[(0.5, 0.9), (0.6, 0.6), 
                                                     (0.9, 0.4), (0.3, 0.9), 
                                                     (0.9, 0.2)], 
                     fontsize = tick_label_sz, inline = True, 
                     use_clabeltext = True, inline_spacing = 10)

# actual
act = ax.plot(act_success_ratio, act_hit_rates, '-', color = 'gold', linewidth = 6, zorder = 1)
ax.scatter(act_success_ratio, act_hit_rates, color = 'gold',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)

# training
tra = ax.plot(tra_success_ratio, tra_hit_rates, '-', color = 'dodgerblue', linewidth = 6, zorder = 1)
ax.scatter(tra_success_ratio, tra_hit_rates, color = 'dodgerblue',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)

# testing
tes = ax.plot(tes_success_ratio, tes_hit_rates, '-', color = 'orangered', linewidth = 6, zorder = 1)
ax.scatter(tes_success_ratio, tes_hit_rates, color = 'orangered',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)

# ERA5
era = ax.plot(era_success_ratio, era_hit_rates, '-', color = 'lime', linewidth = 6, zorder = 1)
ax.scatter(era_success_ratio, era_hit_rates, color = 'lime',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)


# legend for colors
legend_handles = [Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='gold', linestyle='-', linewidth=5),
          Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='dodgerblue', linestyle='-', linewidth=5),
          Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='orangered', linestyle='-', linewidth=5),
          Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='lime', linestyle='-', linewidth=5)]
lgnd = ax.legend(handles = legend_handles, labels = ['Actual OPG', 'Training OPG', 'Testing OPG', 'LI + QM ERA5'], 
          title="Downscaled Precipitation From:", loc = "lower right", 
          prop = {'size': tick_label_sz}, markerscale = 4, ncol=2, framealpha=1,
          alignment='left')
plt.setp(lgnd.get_title(),fontsize=tick_label_sz)


# labels
ax.set_xlabel("Success Ratio (1 - FAR)", fontsize = axis_label_sz)
ax.set_ylabel("Probability of Detection", fontsize = axis_label_sz)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.text(0.3,0.38,"Frequency Bias",fontdict=dict(fontsize=axis_label_sz, rotation=45), zorder = 12)
ax.tick_params(axis = 'both', which = 'major', labelsize = tick_label_sz, pad = 5)

# colorbar
cbar = fig.colorbar(csi_contour, pad = 0.01, ax = ax, orientation = 'vertical', extend = 'neither')
cbar.set_label('Critical Success Index', fontsize = axis_label_sz)
cbar.set_ticks(np.round(np.arange(0, 1.01, 0.1), 1))
cbar.set_ticklabels(np.round(np.arange(0, 1.01, 0.1), 1))
cbar.ax.tick_params(labelsize=tick_label_sz)

path = model_dir + model_name + "/"
# plt.savefig(f"{path}performance_diagram.png", dpi=300, 
#               transparent=True, bbox_inches='tight')

plt.show()





#%% Plot - colorbar on top

fig, (ax, lgnd) = plt.subplots(nrows=2, figsize=(8, 11), facecolor='w', gridspec_kw={"height_ratios":[1, 0.15]})
grid_ticks  = np.arange(0, 1.01, 0.01)
sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
bias        = pod_g / sr_g
csi         = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)

axis_label_sz = 20
tick_label_sz = 16
s = 5
sl = 1

# Plot contours for CSI
csi_contour = ax.contourf(sr_g, pod_g, csi, np.arange(0, 1.01, 0.1), 
                           cmap= "Greys", alpha = 0.825)
csi_contourr = ax.contour(sr_g, pod_g, csi, np.arange(0, 1.01, 0.1), 
                           colors = 'k', linewidths = 0.35)

# Plot Contours for BIAS
b_contour   = ax.contour(sr_g, pod_g, bias, [0.25, 0.5, 1, 2, 4], colors="k", 
                          linestyles="dashed")
clabels = ax.clabel(b_contour, fmt="%1.1f", manual=[(0.5, 0.9), (0.6, 0.6), 
                                                     (0.9, 0.4), (0.3, 0.9), 
                                                     (0.95, 0.2)], 
                     fontsize = tick_label_sz, inline = True, 
                     use_clabeltext = True, inline_spacing = 10)


# actual
act = ax.plot(act_success_ratio, act_hit_rates, '-', color = 'gold', linewidth = 6, zorder = 1)
ax.scatter(act_success_ratio, act_hit_rates, color = 'gold',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)

# training
tra = ax.plot(tra_success_ratio, tra_hit_rates, '-', color = 'dodgerblue', linewidth = 6, zorder = 1)
ax.scatter(tra_success_ratio, tra_hit_rates, color = 'dodgerblue',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)

# testing
tes = ax.plot(tes_success_ratio, tes_hit_rates, '-', color = 'orangered', linewidth = 6, zorder = 1)
ax.scatter(tes_success_ratio, tes_hit_rates, color = 'orangered',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)

# ERA5
era = ax.plot(era_success_ratio, era_hit_rates, '-', color = 'lime', linewidth = 6, zorder = 1)
ax.scatter(era_success_ratio, era_hit_rates, color = 'lime',  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)


# # legend for colors
legend_handles = [Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='gold', linestyle='-', linewidth=5),
          Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='dodgerblue', linestyle='-', linewidth=5),
          Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='orangered', linestyle='-', linewidth=5),
          Line2D([], [], marker='.', markeredgecolor = 'k', markeredgewidth=2, color='lime', linestyle='-', linewidth=5)]
lgnd_color = ax.legend(handles = legend_handles, labels = ['Actual OPG', 'Training OPG', 'Testing OPG', 'LI + QM ERA5'], 
          title="Downscaled Precipitation From:", loc = "lower right", 
          prop = {'size': tick_label_sz}, markerscale = 4, ncol=2, framealpha=0.8,
          alignment='left')
plt.setp(lgnd_color.get_title(),fontsize=tick_label_sz)


# labels
ax.set_xlabel("Success Ratio (1 - FAR)", fontsize = axis_label_sz)
ax.set_ylabel("Probability of Detection", fontsize = axis_label_sz)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.text(0.3,0.38,"Frequency Bias",fontdict=dict(fontsize=axis_label_sz, rotation=45), zorder = 12)
ax.tick_params(axis = 'both', which = 'major', labelsize = tick_label_sz, pad = 5)

# colorbar
cbar = fig.colorbar(csi_contour, pad = 0.01, ax = ax, orientation = 'horizontal', location = 'top', extend = 'neither')
cbar.set_label('Critical Success Index', fontsize = axis_label_sz, labelpad=10)
cbar.set_ticks(np.round(np.arange(0, 1.01, 0.1), 1))
cbar.set_ticklabels(np.round(np.arange(0, 1.01, 0.1), 1))
cbar.ax.tick_params(labelsize=tick_label_sz)


# legend plot
xx = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
yy = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
y_lev = 1.6


# ERA5
lgnd.plot(xx, yy*y_lev*2, '-', color = [0.6, 0.6, 0.6], linewidth = 6, zorder = 1)
lgnd.scatter(xx, yy*y_lev*2, color = [0.6, 0.6, 0.6],  s = [10*s, 25*s, 50*s, 75*s, 95*s], 
            edgecolor = 'k', linewidth = 2, zorder = 10)


# Quantile Labels
lgnd.text(xx[0], yy[0]*y_lev*3.4, "5th", ha='center', fontsize=tick_label_sz)
lgnd.text(xx[1], yy[0]*y_lev*3.4, "25th", ha='center', fontsize=tick_label_sz)
lgnd.text(xx[2], yy[0]*y_lev*3.4, "50th", ha='center', fontsize=tick_label_sz)
lgnd.text(xx[3], yy[0]*y_lev*3.4, "75th", ha='center', fontsize=tick_label_sz)
lgnd.text(xx[4], yy[0]*y_lev*3.4, "95th", ha='center', fontsize=tick_label_sz)
lgnd.text(0.5,  yy[0]*y_lev*5.8, "Precipitation Percentile Threshold:", va='center', 
          ha='center', fontsize=tick_label_sz)

lgnd.set_ylim([0.12, 1.12])
lgnd.set_xlim([0, 1.00])
lgnd.set_xticks([])
lgnd.set_yticks([])


path = model_dir + model_name + "/"
# plt.savefig(f"{path}performance_diagram_lgnd.png", dpi=300, 
#               transparent=True, bbox_inches='tight')

plt.show()






