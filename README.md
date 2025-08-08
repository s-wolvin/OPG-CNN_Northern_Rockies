# OPG-CNN_Northern_Rockies

## Evaluation of a Convolutional Neural Network to Predict Wintertime Orographic Precipitation Gradients of the CONUS Northern Rockies

### Project Description: 

A convolutional neural network (CNN) is presented for downscaling winter precipitation in the western Contiguous United States (CONUS) based on observed orographic precipitation gradients (OPGs). The complex terrain of the western CONUS was divided into topographic facets based on regional terrain orientation, where daily OPGs were formulated from the change in accumulated precipitation with respect to the change in elevation. Observed precipitation was from the Global Historical Climatology Network (GHCN)-Daily from 1979 to 2018. CNN prediction of OPGs on topographic facets allows for continuous spatial prediction of precipitation as the value depends only on the elevation, while conventional encoder-decoder CNN architectures restrict precipitation prediction to the grid spacing of the trained output. 

This repository presents the code corresponding to a CNN test case using OPGs from the Northern Rockies of the western CONUS, predicted by Pacific-North American meteorological fields from ECMWF ERA5 Reanalysis. Evaluation of the CNN included statistical metrics of mean absolute error, mean squared error, and correlation, performance analysis of downscaled precipitation, _k_-means clustering of OPG event types, and composite Gradient-weighted Class Activation Maps (Grad-CAM) for each _k_-means cluster. 

In testing, the CNN accounted for 34% of OPG variance with a mean absolute error of about 2.9 mm km<sup>-1</sup>. Compared to the GHCN-Daily, the overall mean precipitation error from OPG predictions was -0.6 mm, with an interquartile range of 1.7 mm. To evaluate the reasonableness of the variables and regions focused on by the CNN, we applied Gradient-weighted Class Activation Mapping (Grad-CAM) to _k_-means clusters of daily OPG. The Grad-CAM analysis indicated that the CNN focused on physically plausible indicators of OPG for each cluster, such as upstream coastal moisture transport towards the Northern Rockies.

<a href="url"><img src="https://github.com/s-wolvin/OPG-CNN_Northern_Rockies/blob/main/opg_dataset/facet-orienations_ghcnd_northern-rockies.jpeg" align="center" alt="Western CONUS Domain" width="450"></a>


### Datasets:
* [Bohne et al. 2020](https://doi.org/10.1175/JHM-D-19-0229.1) - Climatology of orographic precipitation gradients of the western United States, subsetted to the Northern Utah region of winter (DJF) events from 1979 to 2017.
* [ECMWF ERA5](https://doi.org/10.1002/qj.3803) - Hourly data on pressure levels and single levels from 1940 to present, subsetted to latitudes [36°N, 45°N], longitudes [-119°W, -106°W], of winter (DJF) events from 1979 to 2017. This dataset was accessed through the publicly available Copernicus Climate Change Service (C3S) Climate Data Store (CDS). The ERA5 predictor variables were processed from 6-hourly data on [pressure levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) and 6-hourly data on [single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).

## Getting Started:
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

### 3. Create Your Virtual Environment From The YML Files
The repository contains two environment files, in which the `cnn_gpu.yaml` file requires GPUs for the Tensorflow-GPU python library that trains and loads the CNN model, and _____.yaml does not use GPUs and is used for all scripts with plotting and analysis that do not import Tensorflow-GPU.

From your command prompt, go to the home directory.
```
cd ~
```
The following command creates the needed environment and downloads all required Python libraries in the case of `cnn_gpu.yml`. For the second environment, replace the .yml file name.
```
conda env create -f cnn_gpu.yml
```
Once Anaconda sets up your environment, activate it using the activate function. For example, if using the `cnn_gpu.yaml`, the environment is named `opg_cnn`.
```
conda activate opg_cnn
```
To check if the environment was installed correctly, run the following line.
```
conda env list
```
These environments include the Spyder IDE.


## Folder Structure:
    .
    ├── k-means_clustering_and_Grad-CAM    # Composite Grad-CAMs formulated from k-means clusters
    ├── opg_dataset                        # Original MATLAB files of labeled facets and OPGs
    ├── plt_activation_layers              # Plotting activation maps of the CNN
    ├── plt_eval_precip_prediction         # Plots for evaluating precipitation prediction
    ├── train_cnn                          # Create, train CNN, save output statistics, datasets, and CNN
    ├── README.md                 
    └── environment.yml                    # Environment file

## Acknowledgment

Support for this project was provided by the NOAA NWS Collaborative Science, Technology, and Applied Research (CSTAR) program (Award: NA17NWS4680001), NASA High Mountain Asia Team (Award: 80NSSC20K1594), and the Wilkes Climate Center Seed Grant (Award: 003921). 

