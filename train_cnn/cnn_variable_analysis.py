""" 
Savanna Wolvin
Created: Sep 15th, 2022
Edited: Jun 1st, 2023
    

##### SUMMARY #####
Script file holding the functions to evaluate input variables for the 
Convolutional Neural Network for cnn_regional_facet_MAIN.py

##### FUNCTION LIST ##########################################################
    plot_feature_maps_conv2d_1() - Function to plot feature maps from the input to the 
                            CNN *** Unfinished
    plot_activation_maps() - Function to plot daily class activation maps and 
                                a histogram of OPG prediction error for the CNN


"""
# %% Global Imports

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nclcmaps as ncm
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import postprocess_cnn_output as pp_cnn
import os



            
# %%
""" FUNCTION DEFINITION: plot_grad_cam
    INPUTS
    save_dir    - Directory to Save the Figures
    model       - Fitted ConvNet
    opg_type    - Type of OPG Value Used in Training
    name        - If These are Training, Testing, or Validation Dates
    atmos       - Xarray Contianing the Atmospheric Values Used
    opg         - Xarray of OPG Values Used
    
    OUTPUT - Plot Class Activation Maps
"""


def plot_grad_cam(save_dir, model, opg_type, name, atmos, opg):

    # Calculate Size of Data
    atm_sz = dict(atmos.sizes)
    opg_sz = dict(opg.sizes)
    atmos_names = np.array(atmos.data_vars)
    count_channels = 0
    count_on_facet = 0
    for ii in atmos.values():
        if len(ii.dims) == 3:
            count_channels += 1
        elif len(ii.dims) == 2:
            count_on_facet += 1
            atmos_names = np.delete(
                atmos_names, np.argmax(ii.name == atmos_names))

    # pull plotting axis values
    lons = np.array(atmos['lon'])
    lats = np.array(atmos['lat'])
    date = atmos['time']
    loni, lati = np.meshgrid(lons, lats)

    # find size of figure
    x_axis = np.round(np.sqrt(count_channels+1)).astype('int')
    y_axis = np.ceil(np.sqrt(count_channels+1)).astype('int')

    # load regional location
    rgn_bnds = opg.attrs['bounds']
    rgn_lons = [rgn_bnds[2], rgn_bnds[2],
                rgn_bnds[3], rgn_bnds[3], rgn_bnds[2]]
    rgn_lats = [rgn_bnds[0], rgn_bnds[1],
                rgn_bnds[1], rgn_bnds[0], rgn_bnds[0]]

    # pull list of OPG values
    opg_values = np.array(opg['opg'])

    # Plot Presets
    edges = [-15, 15]
    step = 1.5
    x_ticks = [-15, -10, -5, 0, 5, 10, 15]
    x_tick_labels = ["", "-10", "", "0", "", "10", ""]
    units = " (mm/km)"
    max_count = 35
    aspect = 0.6
    shift = 1

    # preset values
    cmap_atmos = ncm.cmap('NCV_blu_red')

    extent = [-150, -100, 25, 57.5]

    # loop through each day
    for dayx in range(0, atm_sz['time']):
        if np.sum(np.abs(opg_values[dayx, :] > 0)) > 0:
            # pull values from that day
            # Create Empty Array for Testing Data
            atmos_test_4D_X = np.zeros(
                (1, atm_sz['lat'], atm_sz['lon'], count_channels))
            atmos_test_OF_X = np.zeros((1, opg_sz['facet_num']*count_on_facet))

            # Pull Variables for Testing
            count_4D = 0
            count_OF = 0
            for var_name, values in atmos.items():
                if len(values.dims) == 3:
                    atmos_test_4D_X[:, :, :, count_4D] = np.array(
                        values[dayx, :, :])
                    count_4D += 1
                elif len(values.dims) == 2:
                    atmos_test_OF_X[:, opg_sz['facet_num']*count_OF:opg_sz['facet_num']
                                    * (count_OF+1)] = np.array(values[dayx, :])
                    count_OF += 1

            # Combine Inputs if On-Facet Data Exists
            if count_on_facet > 0:
                inputs = [atmos_test_4D_X, atmos_test_OF_X]
            else:
                inputs = [atmos_test_4D_X]

            # SECOND ATTEMPT
            heatmap = []

            for output_node in range(opg_sz['facet_num']):
                # Create a model that takes the input image and outputs the last
                # convolutional layer and the output predictions
                grad_model = tf.keras.models.Model([model.inputs],
                                                   [model.get_layer('conv2d_2').output, model.output])

                # Then compute the gradient of the top predicted class for our
                # input image with resepct to the activations of the last conv layer
                with tf.GradientTape() as tape:
                    last_conv_layer_output, preds = grad_model(inputs)
                    class_channel = preds[:, output_node]

                # This is the gradient of the output neuron with regard to the
                # output feature map of the last conv layer
                grads = tape.gradient(class_channel, last_conv_layer_output)

                # Formulates the mean gradient of each feature map, how important is each feature map?
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

                # Reduce to 3 dimentions
                last_conv_layer_output = last_conv_layer_output[0]

                # Multiply each channel of the feature map by how important it is
                heatmapx = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

                # Reduce dimentions
                heatmap.append(np.array(heatmapx))

            heatmap = np.concatenate(heatmap, axis=2)  # combine all maps
            heatmap = np.max(heatmap, axis=2) # create max of all maps
            # create percentile of all maps
            # heatmap = np.percentile(heatmap, 90, axis=2)

            heatmap = heatmap / np.max(heatmap[:])  # Scale to max of 1
            heatmap[heatmap < 0] = 0  # Remove all the negative values

            # interpolate the heatmap grid
            heatmap_size = np.shape(heatmap)
            heatmap_lat = np.linspace(
                np.min(lats), np.max(lats), num=heatmap_size[0])
            heatmap_lon = np.linspace(
                np.min(lons), np.max(lons), num=heatmap_size[1])
            h_lon, h_lat = np.meshgrid(heatmap_lon, heatmap_lat)

            # pull atmosphere
            atmos_data = np.squeeze(inputs[0])

            ###### plot heatmap   ##################################
            datacrs = ccrs.PlateCarree()
            projex = ccrs.Mercator(central_longitude=np.mean(lons))
            chnnl = 0

            # create predicted values, convert values of non-units to mm/km
            pred = model.predict(inputs)
            if opg_type == 1:
                pred = pp_cnn.standardized_to_raw(pred, opg) * 1000
            elif opg_type == 2:
                pred = pp_cnn.normalized_to_raw(pred, opg) * 1000

            # new figure like michaels
            fig, ax = plt.subplots(nrows=y_axis, ncols=x_axis,
                                   figsize=(x_axis*4, y_axis*4),
                                   subplot_kw={'projection': projex})

            ax_count = 1
            for row in range(y_axis):
                for col in range(x_axis):

                    if ax_count == 1:
                        # Calc Error
                        if opg_type == 0:
                            error = (opg_values[dayx, :] - pred[0, :]) * 1000
                        else:
                            idx = opg_values[dayx, :] != 0
                            error = (opg_values[dayx, idx] - pred[0, idx])

                        me = np.mean(error)

                        # Plot Histogram

                        lower = np.sum(error < edges[0])
                        upper = np.sum(error > edges[1])
                        n, bins, patches = ax[row, col].hist(error,
                                                             bins=np.arange(
                                                                 edges[0]-step, edges[1]+step+step, step),
                                                             edgecolor="white")
                        ax[row, col].set(xlim=(edges[0]-step, edges[1]+step),
                                         ylim=(0, max_count),
                                         xticks=x_ticks,
                                         xticklabels=x_tick_labels,
                                         yticks=range(0, max_count+1, 10))

                        # Add Outliers
                        patches[0].set_height(patches[0].get_height() + lower)
                        patches[0].set_facecolor('k')
                        patches[-1].set_height(patches[-1].get_height() + upper)
                        patches[-1].set_facecolor('k')

                        # Plot mean error
                        ax[row, col].plot([me, me], [0, 80], c='red')
                        if me > 0:
                            ax[row, col].text(me-shift, max_count-6,
                                              "Mean Error:"+"\n" +
                                              str(np.round(me, decimals=2))+units,
                                              c='red', fontsize=12, horizontalalignment='right')
                        else:
                            ax[row, col].text(me+shift, max_count-6,
                                              "Mean Error:"+"\n" +
                                              str(np.round(me, decimals=2))+units,
                                              c='red', fontsize=12, horizontalalignment='left')

                        # Plot settings
                        ax[row, col].set_aspect(aspect)
                        ax[row, col].set_title("OPG Prediction Error",
                                               fontsize=14, weight='bold')
                        ax[row, col].grid(True)
                        ax[row, col].set_ylabel("Count of Facets", fontsize=12)
                        ax[row, col].set_xlabel("Error"+units, fontsize=12)
                        ax[row, col].tick_params(labelsize=12)

                    elif ax_count > (count_channels+1):
                        ax[row, col].axis('off')

                    else:

                        # Add Z-Scored Atmospheric Data
                        atmos_plot = ax[row, col].contourf(loni, lati,
                                                            np.squeeze(
                                                                atmos_data[:, :, chnnl]),
                                                            cmap=cmap_atmos,
                                                            zorder=1,
                                                            levels=np.arange(-6, 6.5, 0.5),
                                                            transform=datacrs, extend='both')

                        # Cartography Features
                        states_provinces = cfeat.NaturalEarthFeature(
                                                category='cultural',
                                                name='admin_1_states_provinces_lines',
                                                scale='50m', facecolor='none')
                        ax[row, col].add_feature(cfeat.COASTLINE.with_scale(
                            '110m'), zorder=2, edgecolor="saddlebrown")
                        ax[row, col].add_feature(cfeat.BORDERS.with_scale(
                            '110m'), zorder=3, edgecolor="saddlebrown")
                        ax[row, col].add_feature(states_provinces, 
                                     zorder=4, edgecolor="saddlebrown")
                        ax[row, col].set_extent(extent)

                        # Add Regional Location
                        ax[row, col].plot(rgn_lons, rgn_lats, linewidth=2,
                                          c='black', zorder=5, transform=datacrs)

                        # Add importance of the atmos data and layer over
                        hx = heatmap.copy()*100
                        hatch = ax[row, col].contourf(h_lon, h_lat, hx, colors='none',
                                              zorder=6, transform=datacrs,
                                              hatches=[None, '..', 'oo','OO'], 
                                              levels=[0, 25, 50, 75, 100])

                        # Add Title
                        ax[row, col].set_title(rename_title(atmos_names[chnnl]),
                                               fontsize=14,
                                               weight='bold')

                        # Add To Channel Count Value
                        chnnl += 1

                    ax_count += 1

            # Adjust Spacing
            #plt.subplots_adjust(wspace=0.25, hspace=0.25)
            
            # Add percentage that are above 80%
            prct = str(np.round((np.sum(heatmap > 0.50) / heatmap.size)*100, decimals=1))
            fig.supxlabel(f"{prct}% of Domain is >50%", x=0.93, y=0.09, horizontalalignment="right")
            
            # Add Colorbar for Atmos
            cbar_atmos = fig.add_axes([0.93, 0.35, 0.03, 0.55])
            cb_at = fig.colorbar(atmos_plot, cax=cbar_atmos, ticks=np.arange(-6,8,2),
                                  pad=0.0, aspect=15, fraction=0.032)
            cb_at.set_label('Z-Score of Atmospheric Data', size=16)
            cb_at.ax.tick_params(labelsize=16)

            # Add Colorbar for Hatching
            cbar_cam = fig.add_axes([0.93, 0.12, 0.03, 0.2])
            cb_cam = fig.colorbar(hatch, cax=cbar_cam,
                                  pad=0.0, aspect=15, fraction=0.032)
            cb_cam.set_label("Max. Grad-CAM (%)", size=16)
            cb_cam.ax.tick_params(labelsize=16)
            
            # Add Title with Date and Actual/Predicted Values
            df = np.array(date[dayx], dtype='str')
            df = str(df)
            plt.suptitle(df[0:10], fontsize=18, y=0.93, weight='bold')

            # create path to save
            path = save_dir + "activ_map/"
            if os.path.exists(path) == False:
                os.mkdir(path)

            plt.savefig(path + df[0:10] + "_" + name + ".png", dpi=400, transparent=True,
                        bbox_inches='tight')

            plt.close()
            
            # Show Figure
            # plt.show()




#%%
""" FUNCTION DEFINITION: rename_title
    INPUTS
    var_name    - Name of variable 
    
    OUTPUT - Better Variable Name for a Title
"""


def rename_title(var_name):
    
    name = var_name[:-3]
    level = var_name[-3:]
    
    # set begining of title
    if name == "IVT":       title = "IVT"
    elif name == "hgt":     title = "Geo. Height"
    elif name == "precip":  title = "Accumulated Precipitation"
    elif name == "shum":    title = "Specific Humidity"
    elif name == "temp":    title = "Temperature"
    elif name == "uwnd":    title = "U-Wind"
    elif name == "vwnd":    title = "V-Wind"
    elif name == "vwnd_10m": title = "V-Wind 10-m"
    elif name == "wwnd":    title = "W-Wind"
    
    # Set end of title
    if level != "sfc":
        title = title + " " + level + " hPa"
            
    return title
    

















