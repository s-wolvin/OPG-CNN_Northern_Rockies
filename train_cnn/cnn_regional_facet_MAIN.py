""" 
Savanna Wolvin
Created: Sep 9th, 2022
Edited: Jun 1st, 2023
    

##### SUMMARY ################################################################
This program is designed to pull the desired NCEP Reanalysis data or ECMEF ERA5
data and regional facet OPG timeseries, annd then train a Convolutional Neural 
Network to predict the OPG based on the chosen conditions.


##### INPUT ##################################################################
data_directory (str)    - Directory to the Reanalysis data
data_types (dict)       - Dictionary Containing Desired Atmospheric Variables 
                            to Train the CNN Model. 4 Dimentional Data (time, 
                            lat, lon, pressure) Gets Pulled From the 
                            Latitudinal and Longitudinal Domain Defined Below. 
                            3 Dimentional Data (time, lat, lon) Gets 
                            Interpolated to the Mean of the Specified Facets' 
                            Latitudes and Longitudes
data_set (str)          - Name of Dataset to pull from
                      
#### All Possible Datatypes ##################################################
ECMWF ERA5
#### 1 Dimentional, On-Facet Data:
Int. Vapor Transport - "IVT":       [on-facet]
Total Precipitation  - "precip":    [on-facet]
2-m Temperature      - "temp_2m":   [on-facet]
2-m Dew Point        - "dtemp_2m":  [on-facet]
10-m U-Winds         - "uwnd_10m":  [on-facet]
100-m U-Winds        - "uwnd_100m": [on-facet]
10-m V-Winds         - "vwnd_10m":  [on-facet]
100-m V-Winds        - "vwnd_100m": [on-facet]
Mean Sea-Level Pres. - "mslp":      [on-facet]

#### 2 Dimentional Data:
Geopotential Heights - "hgt":       [1000,925,850,700,600,500,400,300,200]
Potential Temp.      - "pTemp":     [1000,925,850,700,600,500,400,300,200]
Relative Humidity    - "rhum":      [1000,925,850,700,600,500,400,300,200]
Specific Humidity    - "shum":      [1000,925,850,700,600,500,400,300,200]
Temperature          - "temp":      [1000,925,850,700,600,500,400,300,200]
Equ. Poten. Temp.    - "theta-e":   [1000,925,850,700,600,500,400,300,200]
U-Winds              - "uwnd":      [1000,925,850,700,600,500,400,300,200]
V-Winds              - "vwnd":      [1000,925,850,700,600,500,400,300,200]
W-Winds              - "wwnd":      [1000,925,850,700,600,500,400,300,200]
Int. Vapor Transport - "IVT":       [sfc]
Total Precipitation  - "precip":    [sfc]
2-m Temperature      - "temp_2m":   [sfc]
2-m Dew Point        - "dtemp_2m":  [sfc]
10-m U-Winds         - "uwnd_10m":  [sfc]
100-m U-Winds        - "uwnd_100m": [sfc]
10-m V-Winds         - "vwnd_10m":  [sfc]
100-m V-Winds        - "vwnd_100m": [sfc]
Mean Sea-Level Pres. - "mslp":      [sfc]




NCEP/NCAR REANALYSIS-1
#### 1 Dimentional, On-Facet Data:
Precip. Total   - "prate":  [on-facet]
Rel. Humidity   - "rhum":   [on-facet, crest level]
U-Winds         - "uwnd":   [on-facet, crest level]
V-Winds         - "vwnd":   [on-facet, crest level]
W-Winds         - "omega":  [crest level]

#### 2 Dimentional Data:
Precip. Total   - "prate":  [sfc]
IVT             - "IVT":    [all]
Geop. Heights   - "hgt":    [1000,925,850,700,600,500,400,300,250,200]
Rel. Humidity   - "rhum":   [sig995,1000,925,850,700,600,500,400,300]
Spec. Humidity  - "shum":   [1000,925,850,700,600,500,400,300]
Temperature     - "air":    [1000,925,850,700,600,500,400,300,250,200]
U-Winds         - "uwnd":   [10m,1000,925,850,700,600,500,400,300,250,200]
V-Winds         - "vwnd":   [10m,1000,925,850,700,600,500,400,300,250,200]
W-Winds         - "omega":  [1000,925,850,700,600,500,400,300,250,200,150,100]
##############################################################################
#  On Facet IVT, 700 winds, 850 temp, IVT, 500 GPH, 300 winds

# Lucas Bohne
# Mean Precipitation

# Matt DeMaria
# On-Facet Precipitation, 
# Crest-Level Winds (U-Winds)? - figure 2.3
# 700 hPa Relative Humidity for dry layers
# Low-Level Winds
# 850 hPa Temperatures
# Surface RH
# 300 hPa Winds


years (list)            - Start and End Years to Train the CNN Model
latitudes (list)        - Latitudes of the Domain for the Atmospheric 
                            Variables
longitudes (list)       - Longitudes of the Domain for the Atmospheric 
                            Variables
grid_spacing (float)    - Grid-Spacing of the Reanalysis Dataset
months (list)           - Specific Months to Train the CNN Model


prct_test_days (float)  - Percent of Days to Reserve for Testing
prct_vldtn_days (float) - Percent of Days to Reserve for Validation
conv_act_func (str)     - Activation Function for the Convolutions
dense_act_func (str)    - Activation Function for the Dense Layers
dropout_rate (float)    - Percent Dropout before the Dense Layers
kernal_sz (int)         - Kernal Size, The Convolutional Window
kernal_num (int)        - Number of Kernals, Doubled each Subsequent Convolution
epoch_num (int)         - Max Number of Epochs to Train the Neural Network
nn_layer_width (int)    - Number of Nodes in Each Hidden Layer
nn_hidden_layer (int)   - Number of Hidden Layers in the Neural Network
batch_sz (int)          - Size of the Batch Used During Training Epochs 
optimizer_class (str)   - Type of Optimizer used in Training
loss_metric (str)       - Metric Used to Measure Loss
patience (int)          - Number of Epochs With No Improvement to Stop Training


opg_directory (str)     - Directory to the Facet OPGs
facet_directory (str)   - Directory to the Facets
facet_region (str)      - The Regional ID to Identify What Regions Facets to 
                            Predict
opg_nans (int)          - How to Handle NaNs in the Facet OPG Dataset
opg_type (int)          - What Type of OPG Values to Use
prct_obs (float)        - Percent of Obs Desired in the Timeseries to Allow in 
                            Training the CNN


save_dir (str)          - Directory to save the Model and Cache Data
save_model (bool)       - Boolean to Signal to the Program to Save the CNN 
                            Model if it Has Not Been Done Before 
                            https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model
notes (str)             - Extra notes to add to model_info.txt


##### OUTPUT #################################################################
Saved model
Save logistics



"""
#%% to do 
#1 simplify changing the xarray into a np.array
#3 change from 2D convolutions to 3D convolutions
#4 k-fold cross validation

# update load era5 to include crest level values
# normalize the batches


#%% Global Imports ###########################################################

# import tensorflow as tf
# import tensorflow.keras as tf_k
import numpy as np
from datetime import datetime, timedelta
import os
import random as rd
import load_data as ld
import conv_nn as cnn
import cnn_plots as plots
import grad_cam as grd_cm
import feature_maps as ftr_mp
import model_output as output
import terrain_plots as terrain
import pandas as pd




#%% File Paths and Preset Variables ##########################################

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### Atmsopheric Variable data
data_set  = "ERA5" # Reanalysis, ERA5
# Variable Name: Pressure Levels

# data_types      = {"IVT": ["sfc"],
#                     "uwnd": ["700", "300"],
#                     "vwnd": ["700", "300"],
#                     "wwnd": ["700", "300"],
#                     "hgt": ["700", "300"],
#                     "rhum": ["700"],
#                     "uwnd_10m": ["on-facet"],
#                     "vwnd_10m": ["on-facet"],
#                     "precip": ["sfc"]}

###################
data_types      = {"uwnd": ["700"],
                    "vwnd_10m": ["sfc"],
                    "wwnd": ["700"],
                    "hgt": ["500"],
                    "IVT": ["sfc"],
                    "precip": ["sfc"],
                    "shum": ["850"],
                    "temp": ["700"]}

# data_types      = {"uwnd": ["700"],
#                     "IVT": ["sfc"],
#                     "precip": ["sfc"]}

### Domain Values
years           = [1979, 2018]
latitudes       = [25, 60] # [25, 60]
longitudes      = [-150, -100] # Reanalysis [210, 260], ERA5 [-150, -100]
months          = [12, 1, 2]

### CNN Model Values
prct_test_days  = 0.2
prct_vldtn_days = 0.1
conv_act_func   = 'relu'
dense_act_func  = 'sigmoid'
dropout_rate    = 0.2
kernal_sz       = 3
kernal_num      = 32
epoch_num       = 100
nn_layer_width  = 48
nn_hidden_layer = 2
batch_sz        = 32
optimizer_class = "sgd"
loss_metric     = "mean_squared_error"
patience        = 7

### Facet Presets
opg_directory = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/opg/"
facet_directory = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/facets/"
# CR - Coastal Ranges; NR - Northern Rockies; IPN - Inland Pacific Northwest; 
# SW - Southwest; SR - Southern Rockies
facet_region    = "NR"
# How to Handle NaN's? 0 - remove all NaN days, 1 - Set NaNs to zero, 2 - Set NaNs to average
opg_nans        = 1
# Type of OPG Values to Use? 0 - raw OPG, 1 - standardized, 2 - normalized
opg_type        = 1
# Percent of Days Within Timeseries the Facet Must Have Observations
prct_obs        = 0.2
# Minimum number of stations on the Facet to use the OPG value
num_station = 3

### Saving Presets
save_dir = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/cstar/" + \
                                    "regional_facet_cnn_weighting/" + facet_region + "/"
save_model = True

### Notes to add to model_info.txt
notes = "\nNotes: \n" + \
        "- the model output statisitcs have been fixed \n" + \
        "- the oversampling of higher precipitation days was removed \n" + \
        "- percent of obs is 20% FOUND THAT NEGATIVES WERE NOT BEING COUNTED!!!! \n" + \
        "- No weighting \n" + \
        "- Custom loss function to account for NaNs \n" + \
        "- Full runthrough with selected variables \n" + \
        "- Full Run with relu/sigmoid \n" + \
        "- removed facet weighting \n" + \
        "- Re-run with 3 convolutions"

# "- Weighted samples by more widespread events" + \
# "- Weighted samples by daily mean OPG" + \
    
#        "- Weight by number of facets/half the total facets \n" + \
#        "- Weights range from 0.06-1.97 \n" + \

#%% Main Function ############################################################

def main():
    
    ##### Create Domain ######################################################
    data_days, data_years, data_lats, data_lons, data_directory = create_domain_arrays(data_set)
    
    
    ##### Pull OPG Data for the Facet ########################################
    facet_opg = ld.get_regional_opg(data_years, opg_directory, facet_directory, 
                                 facet_region, months, opg_nans, opg_type, 
                                 prct_obs, num_station)    
    
    
    ##### Load in the Atmospheric Data #######################################
    if data_set == "Reanalysis":
        atmosphere = ld.create_reanalysis_dataset(data_years, data_lats, data_lons, 
                                              data_types, data_directory, facet_opg, 
                                              facet_directory, facet_opg.time)
    elif data_set == "ERA5":
        atmosphere, atmos_mean, atmos_stdev = ld.create_era5_dataset(data_years, 
                                                 data_lats, data_lons, data_types, 
                                                 data_directory, facet_opg, 
                                                 facet_directory, facet_opg.time)
    
    
    ##### Define Days for Training, Testing ##################################
    train_days, vldtn_days, test_days = allocate_dataset_months(data_days)
    
    
    ##### Create save location ###########################################
    now = datetime.now()
    timestr = now.strftime('%Y-%m-%d_%H%M')
    path = save_dir + timestr + "/"
    if os.path.isdir(path) == False: os.mkdir(path)
    

    ##### Save Text File of Model Information ################################
    cnn_input = 'OPG NaNs = ' + str(opg_nans) + '\n' + \
                   'OPG Type = ' + str(opg_type) + '\n' + \
                   'Months Used: ' + str(months) + '\n' + \
                   '% Obs Required: ' + str(prct_obs) + '\n' + \
                   'Num of Stations: ' + str(num_station) + '\n' + \
                   'Num of Facets = ' + str(len(facet_opg.facet_num)) + '\n' + \
                   'Num of Days = ' + str(len(facet_opg.time)) + '\n' + \
                   '% Test Days = ' + str(prct_test_days) + '\n' + \
                   '% Val. Days = ' + str(prct_vldtn_days) + '\n' + \
                   'Dataset: ' + str(data_set) + '\n' + \
                   'Conv. Act. Func. = ' + conv_act_func  + '\n' + \
                   'Dense Act. Func. = ' + dense_act_func  + '\n' + \
                   'Dropout Rate = ' + str(dropout_rate) + '\n' + \
                   'Kernal Size = ' + str(kernal_sz) + 'x' + str(kernal_sz) + '\n' + \
                   '# Epochs = ' + str(epoch_num) + '\n' + \
                   'Patience = ' + str(patience) + '\n' + \
                   'NN Layers = ' + str(nn_hidden_layer) + '\n' + \
                   'NN Width = ' + str(nn_layer_width) + '\n' + \
                   'Batch Size = ' + str(batch_sz) + '\n' + \
                   'Optimizer: ' + optimizer_class + '\n' + \
                   'Loss Metric = ' + str(loss_metric) + '\n' + \
                   str(data_types.keys()) + '\n' + \
                   str(data_types.values()) + '\n' + \
                   notes
    model_info  = open(path + "model_info.txt", "w")
    model_info.write(cnn_input)
    model_info.close()
    
    
    ##### Plot the locations of the Facets Available #####################
    terrain.facet_map_labeled(path, facet_directory, facet_opg)
    # terrain.facet_map_terrain(path, facet_directory, facet_opg)
    
    
    ##### Set Up the CNN Structure Model #################################
    model = cnn.structure_cnn(atmosphere, conv_act_func, nn_layer_width, 
                              nn_hidden_layer, dense_act_func, dropout_rate,
                              kernal_sz, kernal_num, facet_opg['facet_num'].size, path)
    
    
    ##### Train CNN Model ################################################    
    model = cnn.train_cnn(model, opg_type, path, epoch_num, patience, batch_sz, 
                          optimizer_class, loss_metric, 
                          atmosphere.isel(time=np.isin(atmosphere.time, train_days)), 
                          facet_opg.isel(time=np.isin(facet_opg.time, train_days)),
                          atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)), 
                          facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)))
    
    
    ##### Post-Training Analysis Plots ###################################
    # # Training
    plots.heatmap(path, model, opg_type, "Training", 
                  atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
                  facet_opg.isel(time=np.isin(facet_opg.time, train_days)))
    # # Training by Facet
    plots.heatmap_by_facet(path, model, opg_type, "Training", 
                  atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
                  facet_opg.isel(time=np.isin(facet_opg.time, train_days)))
    # Plot histogram of r-squared values and MSE values of all facets
    plots.hist_r2_rrank_MAE_slope(path, model, opg_type, "Training", 
                  atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
                  facet_opg.isel(time=np.isin(facet_opg.time, train_days)))
    # Save output statistics
    output.stats(path, model, opg_type, "Training", 
                  atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
                  facet_opg.isel(time=np.isin(facet_opg.time, train_days)))
    
    
    # Validation
    if prct_vldtn_days > 0:
        plots.heatmap(path, model, opg_type, "Validation", 
                      atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)))
    # #     plots.heatmap_by_facet(path, model, opg_type, "Validation", 
    # #                   atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)),
    # #                   facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)))
        plots.hist_r2_rrank_MAE_slope(path, model, opg_type, "Validation", 
                      atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)))
        output.stats(path, model, opg_type, "Validation", 
                      atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)))
    
    
    # Testing
    if prct_test_days > 0:
        plots.heatmap(path, model, opg_type, "Testing", 
                      atmosphere.isel(time=np.isin(atmosphere.time, test_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, test_days)))
        plots.heatmap_by_facet(path, model, opg_type, "Testing", 
                      atmosphere.isel(time=np.isin(atmosphere.time, test_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, test_days)))
        plots.hist_r2_rrank_MAE_slope(path, model, opg_type, "Testing", 
                      atmosphere.isel(time=np.isin(atmosphere.time, test_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, test_days)))
        output.stats(path, model, opg_type, "Testing", 
                      atmosphere.isel(time=np.isin(atmosphere.time, test_days)),
                      facet_opg.isel(time=np.isin(facet_opg.time, test_days)))
        
    
    ##### Save CNN Model #################################################
    if save_model:
        model.save(path + "CNN-OPG_MODEL") # SAVE MODEL
        
        path_data = path + "datasets/"
        if os.path.isdir(path_data) == False: os.mkdir(path_data)
        # SAVE ATMOSPHERIC DATA
        atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)).to_netcdf(path_data + "atmos_validation.nc")
        atmosphere.isel(time=np.isin(atmosphere.time, test_days)).to_netcdf(path_data + "atmos_testing.nc")
        atmosphere.isel(time=np.isin(atmosphere.time, train_days)).to_netcdf(path_data + "atmos_training.nc")
        
        # SAVE ATMOSPHERIC MEANS AND STANDARD DEVIATION
        atmos_mean.to_netcdf(path_data + "atmos_mean.nc")
        atmos_stdev.to_netcdf(path_data + "atmos_standardDeviation.nc")
        
        # SAVE OPG DATA
        facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)).to_netcdf(path_data + "opg_validation.nc")
        facet_opg.isel(time=np.isin(facet_opg.time, test_days)).to_netcdf(path_data + "opg_testing.nc")
        facet_opg.isel(time=np.isin(facet_opg.time, train_days)).to_netcdf(path_data + "opg_training.nc")
    
        
    
    ##### Plot Heat Maps of Class Activation #############################
    # grd_cm.plot_grad_cam(path, model, opg_type, "Validation",
    #           atmosphere.isel(time=np.isin(atmosphere.time, vldtn_days)),
    #           facet_opg.isel(time=np.isin(facet_opg.time, vldtn_days)))
    grd_cm.plot_grad_cam(path, model, opg_type, "Testing",
              atmosphere.isel(time=np.isin(atmosphere.time, test_days)),
              facet_opg.isel(time=np.isin(facet_opg.time, test_days)))
    grd_cm.plot_grad_cam(path, model, opg_type, "Training",
              atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
              facet_opg.isel(time=np.isin(facet_opg.time, train_days)))
    
    
    ##### Plot Feature Maps ##############################################
    ftr_mp.plot_feature_maps_conv2d_1(path, model, opg_type, "Training",
              atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
              facet_opg.isel(time=np.isin(facet_opg.time, train_days)))
    ftr_mp.plot_feature_maps_conv2d_1(path, model, opg_type, "Testing",
              atmosphere.isel(time=np.isin(atmosphere.time, test_days)),
              facet_opg.isel(time=np.isin(facet_opg.time, test_days)))  
    
    
    # ftr_mp.plot_feature_maps_conv2d_2(path, model, opg_type, "Training",
    #           atmosphere.isel(time=np.isin(atmosphere.time, train_days)),
    #           facet_opg.isel(time=np.isin(facet_opg.time, train_days)))   
        
        


#%% Pick Random Months ########################################################

def allocate_dataset_months(data_days):
    # Pull Months and Years
    mth_yr = pd.to_datetime(data_days)
    mth_yr  = np.unique(mth_yr.strftime("%Y-%m"))
    
    # Calc Number of Months for each Subset
    num_months = len(mth_yr)
    num_test_months = np.round(prct_test_days * num_months).astype('int')
    num_vldtn_months = np.round(prct_vldtn_days * num_months).astype('int')
    
    # Shuffle the Months
    shuffled_months = rd.sample(set(mth_yr), len(mth_yr))
    shuffled_months = rd.sample(shuffled_months, len(shuffled_months))
    
    # Pull the Test, Validation, and Train Months
    test_monYr = shuffled_months[0:num_test_months]
    test_days = np.array([i for i in data_days if pd.to_datetime(i).strftime("%Y-%m") 
                          in test_monYr]).astype('datetime64[ns]')
    
    vldtn_monYr = shuffled_months[num_test_months:(num_test_months+num_vldtn_months)]
    vldtn_days = np.array([i for i in data_days if pd.to_datetime(i).strftime("%Y-%m") 
                           in vldtn_monYr]).astype('datetime64[ns]')
    
    train_monYr = shuffled_months[(num_test_months+num_vldtn_months):-1]
    train_days = np.array([i for i in data_days if pd.to_datetime(i).strftime("%Y-%m") 
                           in train_monYr]).astype('datetime64[ns]')
    
    return train_days, vldtn_days, test_days




#%% Pick Random Days ##########################################################

def allocate_dataset_days(data_days):
    # calc number of days for each subset
    num_days = len(data_days)
    num_test_days = np.round(prct_test_days * num_days).astype('int')
    num_vldtn_days = np.round(prct_vldtn_days * num_days).astype('int')
    
    # Shuffle the Days
    data_days = rd.sample(set(data_days), len(data_days))
    
    # Pull the Test, Validation, and Train Days
    test_days  = np.sort(data_days[0:num_test_days])
    vldtn_days  = np.sort(data_days[num_test_days:(num_test_days+num_vldtn_days)])
    train_days = np.sort(data_days[(num_test_days+num_vldtn_days):-1])
    
    return train_days, vldtn_days, test_days




#%% Create the needed dimentional arrays: Lats, Lons, Time Domain ############

def create_domain_arrays(data_set):
    print("Create Domain Arrays...")
    
    if data_set == "Reanalysis":
        data_directory = "/uufs/chpc.utah.edu/common/home/strong-group7/" + \
            "savanna/reanalysis/ncep/daily/"
        grid_spacing   = 2.5
    
    elif data_set == "ERA5":
        data_directory = "/uufs/chpc.utah.edu/common/home/strong-group7/" + \
            "savanna/ecmwf_era5/"
        grid_spacing   = 0.5
        
    else: raise Exception("Dataset Not Listed")
    
    # Create Array of All Desired Years
    data_years = np.arange(years[0], years[1]+1).astype('int')
    
    # Create Array of All Desired Days
    data_days = np.arange(datetime(years[0],1,1), datetime(years[1],3,1), timedelta(days=1)).astype(datetime)
    data_days = np.array([i for i in data_days if i.month in months]).astype('datetime64[ns]')

    # Create Latitude and Longitude Arrays
    data_lats = np.arange(latitudes[0], latitudes[1]+0.5, grid_spacing)
    data_lons = np.arange(longitudes[0], longitudes[1]+0.5, grid_spacing)
    
    return data_days, data_years, data_lats, data_lons, data_directory
        
    
    

#%% Call Main ################################################################

data_types = dict(sorted(data_types.items()))

if __name__== "__main__":
    main()




