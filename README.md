# OPG-CNN_Northern_Rockies

## Evaluation of a Convolutional Neural Network to Predict Wintertime Orographic Precipitation Gradients of the CONUS Northern Rockies

### Project Description: 

A convolutional neural network (CNN) is presented for downscaling winter precipitation in complex terrain based on observed orographic precipitation gradients (OPGs). Modeling OPGs on topographic facets allows for continuous spatial prediction on flexible grids, while conventional encoder-decoder architectures restrict prediction to the grid spacing of the trained output. Using the Northern Rockies of the contiguous United States (CONUS) as a test case, we divided the terrain into topographic facets based on regional terrain orientation, and used linear regression to quantify daily OPGs on each facet. Observed precipitation was based on gauge values from the Global Historical Climatology Network (GHCN)-Daily data set from 1979 to 2018. The CNN predicted daily wintertime OPGs using Pacific-North American meteorological fields from ECMWF ERA5 Reanalysis, and accounted for 34% of OPG variance with a mean absolute error of about 2.9 mm km<sup>-1</sup>. Compared to the GHCN-Daily, the overall mean precipitation error from OPG predictions was -0.6 mm, with an interquartile range of 1.7 mm. To evaluate the reasonableness of the variables and regions focused on by the CNN, we applied Gradient-weighted Class Activation Mapping (Grad-CAM) to _k_-means clusters of daily OPG. The Grad-CAM analysis indicated that the CNN focused on physically plausible indicators of OPG for each cluster, such as upstream coastal moisture transport.

### Datasets:
* [Bohne et al. 2020](https://doi.org/10.1175/JHM-D-19-0229.1) - Climatology of orographic precipitation gradients of the western United States, subsetted to the Northern Utah region of winter (DJF) events from 1979 to 2017.
* [ECMWF ERA5](https://doi.org/10.1002/qj.3803) - Hourly data on pressure levels and single levels from 1940 to present, subsetted to latitudes 36°N – 45°N, longitudes -119°W – -106°W, of winter (DJF) events from 1979 to 2017. This dataset was accessed through the publicly available Copernicus Climate Change Service (C3S) Climate Data Store (CDS). The ERA5 predictor variables were processed from 6-hourly data on [pressure levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) and 6-hourly data on [single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).

# Getting Started:
### 1. Fork the Repository to your GitHub

Navigate to the top-right corner of the page, select Fork.

![image](https://github.com/s-wolvin/OPG-CNN-Northern-Utah-CIROH-Workshop/assets/34422513/6b96d86e-1ebb-4652-b0f8-c37fb46da3ca)

Confirm the details of this page and select Create Fork.

![image](https://github.com/s-wolvin/OPG-CNN-Northern-Utah-CIROH-Workshop/assets/34422513/343220ce-ec44-40be-a712-f21eaa2dbccc)

### 2. Clone Repository to your Machine
Identify a location where you would like to work in a development environment. Using the command prompt, change your working directory to this folder and git clone [OPG-CNN_Northern_Rockies](https://github.com/s-wolvin/OPG-CNN_Northern_Rockies). Or clone using GitHub Desktop -> File -> Clone Repository, and paste the link listed below under the URL tab.
```
git clone https://github.com/YOUR-USERNAME/OPG-CNN_Northern_Rockies
```

### 3. Create Your Virtual Environment From The YML File
From your command prompt, go to the home directory.
```
cd ~
```
The command below creates the needed environment and downloads all required Python libraries. The environment will be named `CNN_env`.
```
conda env create -f environment.yml
```
Once Anaconda sets up your environment, activate it using the activate function.
```
conda activate CNN_env
```
To check if the environment was installed correctly, run the following line.
```
conda env list
```



## Folder Structure:
    .
    ├── k-means_clustering_and_Grad-CAM    # Composite Grad-CAMs formulated from k-means clusters
    ├── plot_conv-pooling_layers           # Plotting feature maps of the CNN
    ├── regional-cnn_VS_GHCND              # Plot precipitation evaluation
    ├── train_cnn                          # Create, train CNN, save output statistics, datasets, and CNN
    ├── opg_dataset                        # Original MATLAB files of labeled facets and OPGs
    ├── README.md                 
    └── environment.yml                    # Environment file

## Acknowledgment

Support for this project was provided by the NOAA NWS Collaborative Science, Technology, and Applied Research (CSTAR) program (Award: NA17NWS4680001), NASA High Mountain Asia Team (Award: 80NSSC20K1594), and the Wilkes Climate Center Seed Grant (Award: 003921). 

