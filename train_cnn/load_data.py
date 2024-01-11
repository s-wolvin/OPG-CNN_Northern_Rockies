""" 
Savanna Wolvin
Created: Sep 15th, 2022
Edited: Jun 1st, 2023
    

##### SUMMARY #####
Script file holding the functions needed to load data for 
cnn_regional_facet_MAIN.py

##### FUNCTION LIST ##########################################################
    create_reanalysis_dataset() - Loads NCEP Reanalysis data, outputs an Xarray
    --> formulate_IVT() - Specific function to formulate IVT
    create_era5_dataset() - Loads ECMWF ERA5 data, outputs an Xarray
    get_facet_opg() - Loads the Facet OPG data for a Specified Region, outputs 
                        an Xarray



"""
#%% Global Imports

import xarray as xr
import numpy as np
import scipy.io as sio
from datetime import timedelta, datetime
import pandas as pd
import scipy.stats as sp_stats
from wrf import interplevel
from tqdm import tqdm
from sklearn.utils import class_weight




#%%
""" FUNCTION DEFINITION: create_reanalysis_dataset
    INPUTS
    d_years         - Start and End Years to Train the CNN Model
    d_lats          - Latitudes of the Domain for the Atmospheric Variables
    d_lons          - Longitudes of the Domain for the Atmospheric Variables
    d_types         - Dictionary Containing Desired Atmospheric Variables to 
                        Train the CNN Model. 4 Dimentional Data (time, lat, 
                        lon, pressure) Gets Pulled From The Latitudinal and 
                        Longitudinal Domain Defined Below. 3 Dimentional Data 
                        (time, lat, lon) Gets Interpolated to the Mean of the 
                        Specified Facets' Latitudes and Longitudes
    d_dir           - Directory to the Reanalysis data
    fi_num          - The Facet ID Number to Train the CNN Model For
    facet_dir       - Directory to the Facet OPGs
    fi_opg_time     - Time Coordinates from the Facet's OPG Datafile
    
    OUTPUT (atmosphere) - Xarray dataset created from the NCEP Reanalysis daily 
                            data. Will contain the coordinates (time, lat, lon) 
                            for pressure surfaces, and constain (time) for 
                            surface and 10-m data.
"""

def create_reanalysis_dataset(d_years, d_lats, d_lons, d_types, 
                              d_dir, fi_opg, facet_dir, fi_opg_time):
    print("Download Reanalysis Data...")
    
    atmos_vars = d_types.keys()
    
    atmosphere = xr.Dataset()
    
    hgt = 0
    
    for atmos_varsX in atmos_vars: # loop through each var
        print(str(atmos_varsX))
        pressure_levels = d_types[atmos_varsX]
        
        atmosphereX = xr.Dataset()

##### TO DOWNLOAD AND FORMULATE IVT ##########################################
        if atmos_varsX == "IVT":
            atmosphere = formulate_IVT(atmosphere, d_years, d_lats, d_lons, 
                                       fi_opg_time, d_dir, atmos_varsX+pressure_levels[0])

        else:
            for press_levX in pressure_levels: # loop through each level
                # Create empty dataset to hold the atmospehric variable
                atmosphereX = xr.Dataset()
##### TO DOWNLOAD ANY ON-FACET REANALYSIS DATA ###############################
                if press_levX == "on-facet":
                    atmosphereX = np.zeros((np.shape(fi_opg_time)[0], 
                                            np.shape(fi_opg['facet_num'])[0]))
                    
                    # Pull Facet Lat Lons
                    facets_file = sio.loadmat(facet_dir + 'newtest_facets_latlon_20')
                    
                    # Access All NC Files and Convert Dataset to an Xarray
                    ncfile = xr.open_mfdataset(d_dir + atmos_varsX + ".*.????.nc").sel(time=fi_opg_time)
                    ncfile = ncfile[atmos_varsX]
                    
                    # Loop through each facet
                    count_fi = 0
                    for fi in fi_opg['facet_num']:
                        # Pull mean of facet lat/lon
                        facet_lat = np.mean(facets_file["facets_latlon"][fi-1,0])
                        facet_lon = np.mean(facets_file["facets_latlon"][fi-1,1]) + 360
                    
                        # Index the Lats, Lons, and Time of the NCfile
                        ncfile_fi = ncfile.interp(lat=facet_lat, lon=facet_lon)
                        atmosphereX[:,count_fi] = np.array(ncfile_fi)
                        
                        count_fi += 1

                    # Standardize the Variable by Facet 
                    atmosphereX = ((atmosphereX - np.mean(atmosphereX,axis=0)) / 
                                   np.std(atmosphereX,axis=0))
                    
                    # Add Variable to the Total Structure
                    atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX, 
                                coords=[fi_opg_time, fi_opg['facet_num']],
                                dims=['time', 'facet_num'])

##### TO DOWNLOAD ANY CREST LEVEL REANALYSIS DATA ############################
                elif press_levX == "crest level":
                    atmosphereX = np.zeros((np.shape(fi_opg_time)[0], 
                                            np.shape(fi_opg['facet_num'])[0]))
                    
                    # Pull values needed
                    facets_file = sio.loadmat(facet_dir + 'newtest_facets_latlon_20')
                    elev = xr.open_dataset(facet_dir + "ETOPO2v2g_f4.nc")
                    elev = elev["z"]
                    
                    # Access All NC Files and Convert Dataset to an Xarray
                    ncfile = xr.open_mfdataset(d_dir + atmos_varsX + ".????.nc").sel(time=fi_opg_time)
                    ncfile = ncfile[atmos_varsX]
                    
                    # Pull height & pressure values
                    if hgt == 0:
                        hgt   = xr.open_mfdataset(d_dir + "hgt.????.nc").sel(time=fi_opg_time)
                        press = np.array(hgt["level"])
                        lats  = np.array(hgt["lat"])
                        lons  = np.array(hgt["lon"])
                        press = np.ones([np.shape(hgt['level'])[0], np.shape(hgt['lat'])[0], 
                                        np.shape(hgt['lon'])[0]])*press[:,np.newaxis,np.newaxis]
                        hgt = np.array(hgt["hgt"].mean(dim=["time"]))
                    
                    # Loop through each facet
                    count_fi = 0
                    for fi in fi_opg['facet_num']:                    
                        # Pull facet lat/lon
                        facet_lat = np.mean(facets_file["facets_latlon"][fi-1,0])
                        facet_lon = np.mean(facets_file["facets_latlon"][fi-1,1])
                        
                        # Calc Max Elev
                        elev_fi = np.array(elev.interp(y=facet_lat, x=facet_lon))
                        fi_press = xr.DataArray(data=interplevel(press, hgt, elev_fi),
                                                coords=[lats, lons], dims=['lat','lon'])       
                        fi_press = fi_press.interp(lon=(facet_lon+360), lat=facet_lat)

                        # Index the Lats, Lons, and Time of the NCfile
                        ncfile_fi = ncfile.interp(lat=facet_lat, lon=facet_lon, 
                                                  level=fi_press)
                        atmosphereX[:, count_fi] = np.array(ncfile_fi)
                        
                        count_fi += 1

                    # Standardize the Variable by Facet 
                    atmosphereX = ((atmosphereX - np.mean(atmosphereX,axis=0)) / 
                                   np.std(atmosphereX,axis=0))
                    
                    # Add Variable to the Total Structure
                    atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX, 
                                coords=[fi_opg_time],
                                dims=['time'])

##### TO DOWNLOAD ANY RAW 4D REANALYSIS DATA #################################
                else:
                    if press_levX == "sfc" or press_levX == "10m" or press_levX == "sig995":
                        # Pull All NC Files
                        ncfile = xr.open_mfdataset(d_dir + atmos_varsX + ".*.????.nc").sel(time=fi_opg_time)
                        
                        # Standardized the Value By Grid Point
                        ncfile[atmos_varsX] = ((ncfile[atmos_varsX] - 
                                                    ncfile[atmos_varsX].mean(dim=["time"])) / 
                                                    ncfile[atmos_varsX].std(dim=["time"]))
                        
                        # Add Variable to the Total Structure
                        atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                                    data=ncfile.get(atmos_varsX), 
                                    coords=[fi_opg_time, ncfile.lat, ncfile.lon],
                                    dims=['time','lat','lon'])
                    else:
                        for data_yearsX in d_years: # loop through each year           
                            # Access the NC File and Convert Dataset to an Xarray
                            ncfile = xr.open_dataset(
                                d_dir + atmos_varsX + "." + str(data_yearsX) + ".nc")
                        
                            # Index the Pressure Level, Lats, Lons, and Time of the NCfile
                            # Based on the Facet_OPG Xarray
                            time_values = fi_opg_time[fi_opg_time.dt.year.isin(data_yearsX)]
                            ncfile = ncfile.sel(level=np.array(press_levX), 
                                                lat=d_lats, lon=d_lons, time=time_values)
                            
                            # Save Atmospheric Variable
                            atmosphereX = xr.merge([atmosphereX, ncfile])
                            
                        # Standardized the Value By Grid Point
                        atmosphereX[atmos_varsX] = ((atmosphereX[atmos_varsX] - 
                                                    atmosphereX[atmos_varsX].mean(dim=["time"])) / 
                                                    atmosphereX[atmos_varsX].std(dim=["time"]))
                        
                        # Add Variable to the Total Structure
                        atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                                    data=atmosphereX.get(atmos_varsX), 
                                    coords=[fi_opg_time, atmosphereX.lat, atmosphereX.lon],
                                    dims=['time','lat','lon'])
        
    return atmosphere

# Formulating Specific Variables for the Above Function ####################

def formulate_IVT(atmosphere, d_years, d_lats, d_lons, fi_opg_time, d_dir, name):
    # Create empty dataset to hold the atmospehric variable
    atmosphereX = xr.Dataset()
    
    for data_name in ["shum", "uwnd", "vwnd"]:
        for data_yearsX in d_years: # loop through each year
            # Access the NC File and Convert Dataset to an Xarray
            ncfile = xr.open_dataset(
                d_dir + data_name + "." + str(data_yearsX) + ".nc")
        
            # Index the Pressure Level, Lats, Lons, and Time of the NCfile
            # Based on the Facet_OPG Xarray
            time_values = fi_opg_time[fi_opg_time.dt.year.isin(data_yearsX)]
            ncfile = ncfile.sel(level=([1000, 925, 850, 700, 600, 500, 400, 300]),
                                lat=d_lats, lon=d_lons, time=time_values)
        
            # Save Atmospheric Variable
            atmosphereX = xr.merge([atmosphereX, ncfile])
        
    # Calculate IVT
    atmos_size = dict(atmosphereX.sizes)
    IVT = np.zeros((atmos_size['time'], atmos_size['lat'], atmos_size['lon']))
    for lev in range(len(atmosphereX.level.values)-1):
        dp = (atmosphereX.level.values[lev] - atmosphereX.level.values[lev+1]) * 100
        IVT = IVT + (atmosphereX.shum.values[:,lev+1,:,:] * 
                     np.sqrt(atmosphereX.uwnd.values[:,lev+1,:,:]**2 + atmosphereX.vwnd.values[:,lev+1,:,:]**2) +
                     atmosphereX.shum.values[:,lev,:,:] * 
                     np.sqrt(atmosphereX.uwnd.values[:,lev,:,:]**2 + atmosphereX.vwnd.values[:,lev,:,:]**2)
                     ) * dp
    
    # Calc IVT
    IVT = IVT * (0.5/9.81)
    
    # Standardize the Variable By Grid Point
    IVT = ((IVT - np.mean(IVT,axis=0)) / np.std(IVT,axis=0))
    
    # Add Variable to the Total Structure
    atmosphere[name] = xr.DataArray(
                data=(IVT), 
                coords=[fi_opg_time, atmosphereX.lat, atmosphereX.lon],
                dims=['time','lat','lon'])
    
    
    return atmosphere

 


#%%
""" FUNCTION DEFINITION: create_era5_dataset
    INPUTS
    d_years (array)     - Start and End Years to Train the CNN Model
    d_lats (array)      - Latitudes of the Domain for the Atmospheric Variables
    d_lons (array)      - Longitudes of the Domain for the Atmospheric Variables
    d_types (dict)      - Dictionary Containing Desired Atmospheric Variables to 
                            Train the CNN Model. 4 Dimentional Data (time, lat, 
                            lon, pressure) Gets Pulled From The Latitudinal and 
                            Longitudinal Domain Defined Below. 3 Dimentional Data 
                            (time, lat, lon) Gets Interpolated to the Mean of the 
                            Specified Facets' Latitudes and Longitudes
    d_dir (str)         - Directory to the ERA5 data
    fi_opg (xarray)     - Xarray dataset created from the Facet OPG data
    facet_dir (str)     - Directory to the Facet OPGs
    fi_opg_time (array) - Time Coordinates from the Facet's OPG Xarray
    
    OUTPUT (atmosphere) - Xarray dataset created from the ECMWF ERA5 daily 
                            data. Will contain the coordinates (time, lat, lon) 
                            for pressure surfaces, and constain (time) for 
                            surface and 10-m data.
"""

def create_era5_dataset(d_years, d_lats, d_lons, d_types, 
                              d_dir, fi_opg, facet_dir, fi_opg_time):
    print("Download ERA5 Data...")    
    atmos_vars = d_types.keys()
    atmosphere = xr.Dataset()
    atmos_mean = xr.Dataset()
    atmos_stdev = xr.Dataset()
    
    hgt = 0
    
    for atmos_varsX in atmos_vars: # loop through each var
        pressure_levels = d_types[atmos_varsX]
        
        for press_levX in pressure_levels: # loop through each level
            # Create empty dataset to hold the atmospehric variable
            atmosphereX = xr.Dataset()
##### TO DOWNLOAD ANY ON-FACET REANALYSIS DATA ###############################
            if press_levX == "on-facet":                
                atmosphereX = np.zeros((np.shape(fi_opg_time)[0], 
                                        np.shape(fi_opg['facet_num'])[0]))
                
                # Pull Facet Lat Lons
                facets_file = sio.loadmat(facet_dir + 'newtest_facets_latlon_20')
                
                # Access All NC Files and Convert Dataset to an Xarray
                ncfile = xr.open_mfdataset(
                    d_dir + "daily/sfc/era5_" + atmos_varsX + "_????_oct-apr_daily.nc").sel(time=fi_opg_time)
                
                ncfile = ncfile[atmos_varsX]
                
                # Loop through each facet
                count_fi = 0
                for fi in tqdm(fi_opg['facet_num'], desc=atmos_varsX+" "+press_levX):
                    # Pull mean of facet lat/lon
                    facet_lat = np.mean(facets_file["facets_latlon"][fi-1,0])
                    facet_lon = np.mean(facets_file["facets_latlon"][fi-1,1])
                
                    # Index the Lats, Lons, and Time of the NCfile
                    ncfile_fi = ncfile.interp(latitude=facet_lat, longitude=facet_lon)
                    atmosphereX[:,count_fi] = np.array(ncfile_fi)
                    
                    count_fi += 1

                # Save Standardization Values
                atmos_mean[atmos_varsX+press_levX]=xr.DaraArray(
                            data=np.mean(atmosphereX,axis=0),
                            coords=fi_opg['facet_num'],
                            dims=['facet_num'])
                atmos_stdev[atmos_varsX+press_levX]=xr.DaraArray(
                            data=np.std(atmosphereX,axis=0),
                            coords=fi_opg['facet_num'],
                            dims=['facet_num'])
                
                # Standardize the Variable by Facet 
                atmosphereX = ((atmosphereX - np.mean(atmosphereX,axis=0)) / 
                               np.std(atmosphereX,axis=0))
                
                # Add Variable to the Total Structure
                atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                            data=atmosphereX, 
                            coords=[fi_opg_time, fi_opg['facet_num']],
                            dims=['time', 'facet_num'])
                
                # # Save Standardization Values
                # atmos_mean[atmos_varsX+press_levX]=xr.DaraArray(
                #             data=np.mean(atmosphereX,axis=0),
                #             coords=fi_opg['facet_num'],
                #             dims=['facet_num'])
                # atmos_stdev[atmos_varsX+press_levX]=xr.DaraArray(
                #             data=np.std(atmosphereX,axis=0),
                #             coords=fi_opg['facet_num'],
                #             dims=['facet_num'])

##### TO DOWNLOAD ANY CREST LEVEL REANALYSIS DATA ############################
            elif press_levX == "crest level":
                print("unfinished crest level")

##### TO DOWNLOAD ANY RAW 4D REANALYSIS DATA #################################
            else:
                if press_levX == "sfc":
                    for data_yearsX in tqdm(d_years, desc=atmos_varsX+" "+press_levX): # loop through each year 
                        # Access the NC File and Convert Dataset to an Xarray
                        ncfile = xr.open_dataset(
                            d_dir + "daily/sfc/era5_" + atmos_varsX + "_" + str(data_yearsX) + "_oct-apr_daily.nc")
                        
                        # Index the Pressure Lats, Lons, and Time of the NCfile
                        # Based on the Facet_OPG Xarray
                        time_values = fi_opg_time[fi_opg_time.dt.year.isin(data_yearsX)]
                        ncfile = ncfile.sel(latitude=d_lats, longitude=d_lons, 
                                            time=time_values)
                        
                        # Save Atmospheric Variable
                        atmosphereX = xr.merge([atmosphereX, ncfile])
                    
                    # Save Standardization Values
                    atmos_mean[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX[atmos_varsX].mean(dim=["time"]), 
                                coords=[atmosphereX.latitude, atmosphereX.longitude],
                                dims=['lat','lon'])
                    atmos_stdev[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX[atmos_varsX].std(dim=["time"]), 
                                coords=[atmosphereX.latitude, atmosphereX.longitude],
                                dims=['lat','lon'])
                    
                    # Standardized the Value By Grid Point
                    atmosphereX[atmos_varsX] = ((atmosphereX[atmos_varsX] - atmosphereX[atmos_varsX].mean(dim=["time"])) / atmosphereX[atmos_varsX].std(dim=["time"]))
                    
                    # Add Variable to the Total Structure
                    atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX.get(atmos_varsX), 
                                coords=[fi_opg_time, atmosphereX.latitude, atmosphereX.longitude],
                                dims=['time','lat','lon'])
                    
                    # # Save Standardization Values
                    # atmos_mean[atmos_varsX+press_levX] = xr.DataArray(
                    #             data=atmosphereX[atmos_varsX].mean(dim=["time"]), 
                    #             coords=[atmosphereX.latitude, atmosphereX.longitude],
                    #             dims=['lat','lon'])
                    # atmos_stdev[atmos_varsX+press_levX] = xr.DataArray(
                    #             data=atmosphereX[atmos_varsX].std(dim=["time"]), 
                    #             coords=[atmosphereX.latitude, atmosphereX.longitude],
                    #             dims=['lat','lon'])
                    
                    
                else:
                    for data_yearsX in tqdm(d_years, desc=atmos_varsX+" "+press_levX): # loop through each year           
                        # Access the NC File and Convert Dataset to an Xarray
                        ncfile = xr.open_dataset(
                            d_dir + "daily/press/era5_" + atmos_varsX + "_" + str(data_yearsX) + "_oct-apr_daily.nc")
                    
                        # Index the Pressure Level, Lats, Lons, and Time of the NCfile
                        # Based on the Facet_OPG Xarray
                        time_values = fi_opg_time[fi_opg_time.dt.year.isin(data_yearsX)]
                        ncfile = ncfile.sel(level=int(press_levX), 
                                            latitude=d_lats, longitude=d_lons, 
                                            time=time_values)
                        
                        # Save Atmospheric Variable
                        atmosphereX = xr.merge([atmosphereX, ncfile])
                    
                    # Save Standardization Values
                    atmos_mean[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX[atmos_varsX].mean(dim=["time"]), 
                                coords=[atmosphereX.latitude, atmosphereX.longitude],
                                dims=['lat','lon'])
                    atmos_stdev[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX[atmos_varsX].std(dim=["time"]), 
                                coords=[atmosphereX.latitude, atmosphereX.longitude],
                                dims=['lat','lon'])
                    
                    # Standardized the Value By Grid Point
                    atmosphereX[atmos_varsX] = ((atmosphereX[atmos_varsX] - atmosphereX[atmos_varsX].mean(dim=["time"])) / atmosphereX[atmos_varsX].std(dim=["time"]))
                    
                    # Add Variable to the Total Structure
                    atmosphere[atmos_varsX+press_levX] = xr.DataArray(
                                data=atmosphereX.get(atmos_varsX), 
                                coords=[fi_opg_time, atmosphereX.latitude, atmosphereX.longitude],
                                dims=['time','lat','lon'])
                    
                    # # Save Standardization Values
                    # atmos_mean[atmos_varsX+press_levX] = xr.DataArray(
                    #             data=atmosphereX[atmos_varsX].mean(dim=["time"]), 
                    #             coords=[atmosphereX.latitude, atmosphereX.longitude],
                    #             dims=['lat','lon'])
                    # atmos_stdev[atmos_varsX+press_levX] = xr.DataArray(
                    #             data=atmosphereX[atmos_varsX].std(dim=["time"]), 
                    #             coords=[atmosphereX.latitude, atmosphereX.longitude],
                    #             dims=['lat','lon'])
                    
                    
    # When all values in a timeseries is zero, the values becomes NAN, so here
    # I am replacing all nan values with 0
    atmosphere = atmosphere.fillna(0)
        
    return atmosphere, atmos_mean, atmos_stdev
    
    
    
    
#%% 
""" FUNCTION DEFINITION: get_regional_opg
    INPUTS
    d_years (array)     - Array of Years to Pull the OPG Data
    opg_dir (str)       - Directory to the Facet OPGs
    fi_dir (str)        - Direcotry to the Facet Characteristics
    fi_region (str)     - Identification string to the Facet Region Desired
    months (list)       - Months of the Year to Pull the OPG Data
    opg_nans (int)      - Identifier for How to Deal With NaNs Within the OPG 
                            Timeseries
    opg_type (int)      - Identifier for What Type of OPG Values to Use
    prct_obs (float)    - Percent of Non-zero Values Required for Training
    num_station (int)   - Minimum Number of Stations to Use Facet Obs
    
    OUTPUT (facet_opg) - Xarray dataset created from the Facet OPG data. Will 
                            contain the coordinates (time).
"""

def get_regional_opg(d_years, opg_dir, fi_dir, fi_region, months, opg_nans, 
                     opg_type, prct_obs, num_station):
    print("Pull Regional OPG Obs...")
    
    # load lats/lons/facets/orientation
    mat_file = sio.loadmat(fi_dir + 'lats')
    lats  = mat_file['lats']
    
    mat_file = sio.loadmat(fi_dir + 'lons')
    lons  = mat_file['lons']
    
    mat_file = sio.loadmat(fi_dir + 'facets_labeled')
    facets  = mat_file['facets_i']
    
    mat_file = sio.loadmat(fi_dir + 'facets')
    orientation  = mat_file['facets']
    
    # calculate the mean lat/lons
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

    print("Download Facet OPG Data...")
    time_ref    = np.arange(datetime(1979, 1, 1), datetime(2018, 4, 1), 
                                timedelta(days=1), dtype='object')
    
    # Pull Facet OPG
    if opg_type == 0:
        mat_file = sio.loadmat(opg_dir + 'all_opg')  
        facet_opg   = xr.Dataset()
        facet_opg['opg'] = xr.DataArray(data=mat_file['allOPG_qc2'], 
                                        coords=[time_ref, fi_num], 
                                        dims=['time','facet_num'])
    elif opg_type == 1: # Load OPG Mean, STD, and Z-Score
        facet_opg = xr.open_dataset(opg_dir + "OPG_mean_std_z-score.nc")
        facet_opg = facet_opg.rename({"z-score_opg": "opg"})
        
    elif opg_type == 2: # Load OPG Normalization, Min, and Max
        facet_opg = xr.open_dataset(opg_dir + "OPG_min_max_norm.nc")
        facet_opg = facet_opg.rename({"norm_opg": "opg"})
        
    else: raise Exception("OPG Type Not Valid")

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
    
    # Formulate Sample Weights
    # 0-1 weighting by number of facets
    # opgs = facet_opg.opg.values.copy()
    # opgs[opgs == 0] = np.nan
    # y_ = np.sum(~pd.isna(opgs), axis = 1)
    # samp_weight = y_ / np.shape(facet_opg.facet_num.values)[0] # weight by number of facets
    # facet_opg['samp_weight'] = xr.DataArray(data=samp_weight, dims="time")
    
    opgs = facet_opg.opg.values.copy()
    opgs[opgs == 0] = np.nan
    y_ = np.sum(~pd.isna(opgs), axis = 1)
    samp_weight = y_ / (np.shape(facet_opg.facet_num.values)[0]/2) # weight by number of facets
    facet_opg['samp_weight'] = xr.DataArray(data=samp_weight, dims="time")
    
    # Weight by proportional high OPG
    # opgs = facet_opg.opg.values.copy()
    # opgs[opgs == 0] = np.nan
    # y_ = np.nanmean(np.abs(opgs), axis = 1)
    # samp_weight = y_ / np.nanmean(y_)
    # facet_opg['samp_weight'] = xr.DataArray(data=samp_weight, dims="time")
    
    # y_ = np.sum(pd.isna(facet_opg.opg.values), axis=1)  # More Facets, More Weight
    # y_ = np.sum(~pd.isna(facet_opg.opg.values), axis=1) # Less Facets, More Weight
    # samp_weight = class_weight.compute_sample_weight(class_weight='balanced', y=y_)    
    
    

    
    ##### Formulate Case Weights    
    # Weight by percent of Obs
    y_ = np.sum(pd.isna(facet_opg.opg.values), axis=0)/np.mean(np.sum(pd.isna(facet_opg.opg.values), axis=0)) # Less Observations, More Weight
    # y_ = (np.sum(pd.isna(facet_opg.opg.values), axis=0)/np.mean(np.sum(pd.isna(facet_opg.opg.values), axis=0)))**3 # Less Observations, More Weight
    # y_ = np.sum(~pd.isna(facet_opg.opg.values), axis=0)/np.mean(np.sum(~pd.isna(facet_opg.opg.values), axis=0)) # More Observations, More Weight
    # y_ = (np.sum(~pd.isna(facet_opg.opg.values), axis=0)/np.mean(np.sum(~pd.isna(facet_opg.opg.values), axis=0)))**5 # More Observations, More Weight
    
    facet_opg['clss_weight'] = xr.DataArray(data=y_, dims="facet_num")

    
    # Decide How To Handle NaNs
    if opg_nans == 0: ##### Remove Dates With NaN Values #####################
        real_opg_dates = facet_opg.time[~pd.isna(facet_opg.opg.values)]
        facet_opg = facet_opg.where(facet_opg.time==real_opg_dates)
    elif opg_nans == 1: ##### Set All NaNs to Zero ###########################
        facet_opg.opg.values[pd.isna(facet_opg.opg.values)] = 0
    elif opg_nans == 2: ##### Set All NaNs to the Average Value ##############
        facet_opg.opg.values[pd.isna(facet_opg.opg.values)] = facet_opg.opg.mean()
        
    # Display number of OPG days and cancel script if none are found
    print(str(facet_opg.time.size) + " OPG Days from Region " + fi_region)
    print(str(facet_opg.facet_num.size) + " Facets from Region " + fi_region)
    if facet_opg.opg.size == 0: raise Exception("No Precipitation Days Found") 
    
    return facet_opg



