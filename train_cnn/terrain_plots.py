""" 
Savanna Wolvin
Created: Feb 14th, 2023
Edited: Feb 23rd, 2023
    

##### SUMMARY #####
Script file holding the functions to plot the facets and terrain.

##### FUNCTION LIST ##########################################################
    facet_map_terrain() - Plot a map of the Facet region with Facets used in 
                            training shaded by orientation and Facets unused 
                            shaded by elevation
    facet_map_labeled() - Plot a map of the Facet region with Facets used in 
                            training shaded by oreientation and Facets unused 
                            shaded in red



"""
#%% Global Imports

import numpy as np
import matplotlib.pyplot as plt
import nclcmaps as ncm
import scipy.io as sio
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from matplotlib.colors import ListedColormap




#%% 
""" FUNCTION DEFINITION: facet_map_terrain
    INPUTS
    save_dir    - Directory to Save the Figure
    fi_dir      - Directory to the Facet Data
    facet_opg   - Xarray containing OPG data
    
    OUTPUT - Map Plot Shading the Used Facets and Unused Facets
"""

def facet_map_terrain(save_dir, fi_dir, facet_opg):
    cmap_fi = ListedColormap([[0.0, 0.0, 0.0],[0.1428, 0.1428, 0.1428],
              [0.2857, 0.2857, 0.2857],[0.4285, 0.4285, 0.4285],
              [0.5714, 0.5714, 0.5714],[0.7142, 0.7142, 0.7142],
              [0.8571, 0.8571, 0.8571],[1.0, 1.0, 1.0]])
    
    # Preset Values
    fi_nums = facet_opg.facet_num.values
    bounds = facet_opg.bounds
    
    # load lats/lons/facets/orientation
    mat_file = sio.loadmat(fi_dir + 'lats')
    lats  = mat_file['lats']
    mat_file = sio.loadmat(fi_dir + 'lons')
    lons  = mat_file['lons']
    
    mat_file = sio.loadmat(fi_dir + 'elev')
    elev  = mat_file['elev']
    
    mat_file = sio.loadmat(fi_dir + 'facets')
    orientation  = np.array(mat_file['facets']).astype('float')
    mat_file = sio.loadmat(fi_dir + 'facets_labeled')
    facets  = mat_file['facets_i']
    
    # Create Facet Matrix for Plotting
    orix = orientation.astype('float')
    for fi in range(1, np.max(facets)+1):
        if fi in fi_nums: orix[facets == fi] = np.mean(orientation[facets==fi])
        else:
            if np.mean(orientation[facets==fi]) == 9.0: orix[facets == fi] = np.nan
            else: orix[facets == fi] = 9.0
    
    # Create Figure
    datacrs = ccrs.PlateCarree()
    fig = plt.figure( figsize = (8, 8))
    ax = fig.add_axes( [0.1, 0.1, 0.8, 0.8], 
                      projection = ccrs.Mercator(central_longitude=np.mean(lons)))
    
    # Add Facets
    pcm = ax.pcolormesh(lons, lats, orix, cmap=cmap_fi, transform=datacrs, shading='auto')
    pcm.set_clim(0.5,8.5)
    
    # Add Regional Box
    plt.plot([bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]],
              [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]],
              transform=datacrs, color='black')
    
    # Add Terrain of Unused Facets
    cnf = ax.contourf(lons, lats, elev, cmap=ncm.cmap('OceanLakeLandSnow'), 
                  levels=range(0,3800,100), extend='max', transform=datacrs)
    
    # Cartography
    ax.add_feature(cfeat.LAND, facecolor="burlywood")
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="saddlebrown")
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="saddlebrown")
    ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="saddlebrown")
    ax.set_extent([bounds[2]-1, bounds[3]+1, bounds[0]-1, bounds[1]+1])

    # Add Orientation Colorbar
    cbar = plt.colorbar(pcm, ticks=[0,1,2,3,4,5,6,7,8], pad=0.01, fraction=0.75)
    cbar.set_label("Orientation", fontsize=12)
    cbar.ax.set_yticklabels(['','WSW','SSW','SSE','ESE','ENE','NNE','NNW','WNW'])
    cbar.ax.tick_params(labelsize=12)
    
    # Add Elevation Colorbar
    cbar2 = plt.colorbar(cnf, extend='max', location='left', pad=0.01, fraction=0.75)
    cbar2.set_label("Elevation (meters)", fontsize=12)
    cbar2.ax.tick_params(labelsize=12)

    # Save and Show Figure
    print("Save Figure of Facets and Terrain...")
    plt.savefig(save_dir + "facet_map_terrain.png", dpi=400, transparent=True, \
                bbox_inches='tight')

    plt.close()
    # plt.show()
    
    
    
    
#%% 
""" FUNCTION DEFINITION: facet_map_labeled
    INPUTS
    save_dir    - Directory to Save the Figure
    fi_dir      - Directory to the Facet Data
    facet_opg   - Xarray containing OPG data
    
    OUTPUT - Map Plot Shading the Used Facets and Unused Facets
"""

def facet_map_labeled(save_dir, fi_dir, facet_opg):
    cmap_fi = ListedColormap([[0.0, 0.0, 0.0],[0.1428, 0.1428, 0.1428],
              [0.2857, 0.2857, 0.2857],[0.4285, 0.4285, 0.4285],
              [0.5714, 0.5714, 0.5714],[0.7142, 0.7142, 0.7142],
              [0.8571, 0.8571, 0.8571],[1.0, 1.0, 1.0],[0.803,0.380,0.380]])
    
    # Preset Values
    fi_nums = facet_opg.facet_num.values
    bounds = facet_opg.bounds
    
    # load lats/lons/facets/orientation
    mat_file = sio.loadmat(fi_dir + 'lats')
    lats  = mat_file['lats']
    mat_file = sio.loadmat(fi_dir + 'lons')
    lons  = mat_file['lons']
    mat_file = sio.loadmat(fi_dir + 'facets')
    orientation  = np.array(mat_file['facets']).astype('float')
    mat_file = sio.loadmat(fi_dir + 'facets_labeled')
    facets  = mat_file['facets_i']
    
    # Create Facet Matrix for Plotting
    orix = orientation.astype('float')
    for fi in range(1, np.max(facets)+1):
        if fi in fi_nums: orix[facets == fi] = np.mean(orientation[facets==fi])
        else:
            if np.mean(orientation[facets==fi]) == 9.0: orix[facets == fi] = np.nan
            else: orix[facets == fi] = 9.0
    
    # Create Figure
    datacrs = ccrs.PlateCarree()
    fig = plt.figure( figsize = (10, 6))
    ax = fig.add_axes( [0.1, 0.1, 0.8, 0.8], 
                      projection = ccrs.Mercator(central_longitude=np.mean(lons)))
    
    # Add Facets
    pcm = ax.pcolormesh(lons, lats, orix, cmap=cmap_fi, transform=datacrs, shading='auto')
    pcm.set_clim(0.5,9.5)
    
    # Add Regional Box
    plt.plot([bounds[2], bounds[3], bounds[3], bounds[2], bounds[2]],
              [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]],
              transform=datacrs, color='black')
    
    # Add facet labels
    for fi in fi_nums:
        fi_lon = np.median(lons[facets==fi])
        fi_lat = np.median(lats[facets==fi])
        plt.text(fi_lon, fi_lat, str(fi), fontsize = 5, color = 'midnightblue', 
                 transform = datacrs, ha = 'center', va = 'center', 
                 bbox=dict(boxstyle="round", fc=(1,1,1)))
    
    # Cartography
    ax.add_feature(cfeat.LAND, facecolor="burlywood")
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), edgecolor="saddlebrown")
    ax.add_feature(cfeat.BORDERS.with_scale('50m'), edgecolor="saddlebrown")
    ax.add_feature(cfeat.STATES.with_scale('50m'), edgecolor="saddlebrown")
    ax.set_extent([bounds[2]-1, bounds[3]+1, bounds[0]-1, bounds[1]+1])

    # Add colorbar
    cbar = plt.colorbar(pcm, ticks=[0,1,2,3,4,5,6,7,8,9], pad=0.01)
    cbar.set_label("Orientation", fontsize=12)
    cbar.ax.set_yticklabels(['','WSW','SSW','SSE','ESE','ENE','NNE','NNW','WNW','Removed'])
    cbar.ax.tick_params(labelsize=12)

    # Save and Show Figure
    print("Save Figure of Labeled Facets...")
    plt.savefig(save_dir + "facet_map_labeled.png", dpi=400, transparent=True, \
                bbox_inches='tight')

    plt.close()
    # plt.show()
    
    
    
