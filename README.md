# OPG-CNN_Northern_Rockies

### Evaluation of a Convolutional Neural Network to Predict Wintertime Orographic Precipitation Gradients of the CONUS Northern Rockies

Abstract: 

A convolutional neural network (CNN) is presented for downscaling winter precipitation in complex terrain based on orographic precipitation gradients (OPGs). Modeling OPGs on topographic facets allows for continuous spatial prediction on flexible grids, while conventional encoder-decoder architectures restrict prediction to the grid spacing of the trained output. Using the Northern Rockies of the contiguous United States (CONUS) as a test case, we divided the terrain into topographic facets based on regional terrain orientation, and used linear regression to quantify daily OPGs on each facet. Observed precipitation was based on gauge values from the Global Historical Climatology Network (GHCN)-Daily data set from 1979 to 2018. The CNN predicted daily wintertime OPGs using Pacific-North American meteorological fields from ECMWF ERA5 Reanalysis, and accounted for 34\% of OPG variance with a mean absolute error of about 2.9 mm km$^-1$. Compared to the GHCN-Daily, the overall mean precipitation error from OPG predictions was -0.6 mm, with an interquartile range of 1.7 mm. To evaluate the reasonableness of the variables and regions focused on by the CNN, we applied Gradient-weighted Class Activation Mapping (Grad-CAM) to $k$-means clusters of daily OPG. The Grad-CAM analysis indicated that the CNN focused on physically plausible indicators of OPG for each cluster, such as upstream coastal moisture transport.

Support for this project was provided by the NOAA NWS Collaborative Science, Technology, and Applied Research (CSTAR) program. 


# Description of Each Folder:
### $k$-means_clustering_and_Grad-CAM
Contains the Python scripts for $k$-means clustering of regional OPG events of the Western CONUS and creating composite Grad-CAMs for the CNN.

### plot_conv-pooling_layers
Contains the Python script to plot the feature maps of all input, convolutional, and max-pooling layers of a trained CNN model for each sample.

### regional-cnn_VS_GHCND
Contains the Python script to plot the mean precipitation errors of observed, trained predicted, and tested predicted OPGs compared to GHCN-Daily data.

### train_cnn
Contains the Python scripts to create the structure and train a convolutional neural network. It includes saving plots and NetCDF files of output statistics, the training/testing/validation subsets of data, and CNN model. 
