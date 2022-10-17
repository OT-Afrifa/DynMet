                        ################################################################################
                                            ### IMPORT NECESSARY PACKAGES ###
                        ################################################################################

import xarray as xr
import numpy as np
from glob import glob
import os; import dask

import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt


                        ################################################################################
                                            ### Data Reader Functions ###
                        ################################################################################
            
### 1. A long one I made use of 'if' conditions to group like data variables before reading ###
def DataReader_l(data_vars, yr_start, yr_end, dir_path, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    array0 = [];  array1 = []; array2 = []; array3 = [];array4=[]; var_dict = {} 
    for i, variable in enumerate(data_vars):
        for j, year in enumerate([x for x in range(yr_start, yr_end+1)]):
                d_file = glob(os.path.join(dir_path+str(year)+'/*_'+variable+'.*.nc'))[0]
                dask.config.set(**{'array.slicing.split_large_chunks': False})
                if i==0:
                    array0.append(d_file)
                    data = xr.open_mfdataset(array0,combine='by_coords')
                    data = data.sel(longitude = slice(lon_min,lon_max), latitude = slice(lat_max, lat_min))
                    var_dict[variable]=data
                elif i==1:
                    array1.append(d_file)
                    data = xr.open_mfdataset(array1,combine='by_coords')
                    data = data.sel(longitude = slice(lon_min,lon_max), latitude = slice(lat_max, lat_min))
                    var_dict[variable]=data
                elif i==2:
                    array2.append(d_file)
                    data = xr.open_mfdataset(array2,combine='by_coords')
                    data = data.sel(longitude = slice(lon_min,lon_max), latitude = slice(lat_max, lat_min))
                    var_dict[variable]=data
                elif i ==3:
                    array3.append(d_file)
                    data = xr.open_mfdataset(array3,combine='by_coords')
                    data = data.sel(longitude = slice(lon_min,lon_max), latitude = slice(lat_max, lat_min))
                    var_dict[variable]=data
                else:
                    array4.append(d_file)
                    data = xr.open_mfdataset(array4,combine='by_coords')
                    data = data.sel(longitude = slice(lon_min,lon_max), latitude = slice(lat_max, lat_min))
                    var_dict[variable]=data
                    
                    
    for i, variable in enumerate(data_vars):
        var_dict[variable] = var_dict[variable].load()
    return var_dict


"""
2.
A short one that first reads all data variables together as a single dataset before 
selecting each variable (now DataArray), converts them into separate datasets
and store in the dictionary var_dict

"""
def DataReader_s(data_vars, yr_start, yr_end, dir_path, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    array = [];  var_dict = {}; 
    for i, variable in enumerate(data_vars):
        for j, year in enumerate([x for x in range(yr_start, yr_end+1)]):
                d_file = glob(os.path.join(dir_path+str(year)+'/*_'+variable+'.*.nc'))[0]
                array.append(d_file)
                dask.config.set(**{'array.slicing.split_large_chunks': False})
                data = xr.open_mfdataset(array,combine='by_coords')
                data = data.sel(longitude = slice(lon_min,lon_max), latitude = slice(lat_max, lat_min))
                #data = data.load()
                
                if variable[0].isdigit()==True:
                    var_dict[variable] = data[('VAR_'+variable).upper()].to_dataset(promote_attrs=True)
                else:
                    var_dict[variable] = data[variable.upper()].to_dataset(promote_attrs=True)
                    
    for i, variable in enumerate(data_vars):
        var_dict[variable] = var_dict[variable].load()
    return var_dict



                        ################################################################################
                                ### FUNCTIONS FOR PLOTS ###
                        ################################################################################

### Define a function that generates contour filled trend plot using the  cartopy PlateCarree projection ###
def contourfPlot(data, color_map, title, cbar_label,filename, levels=None):
    fig = plt.figure(figsize=(12,8))
    fig.patch.set_facecolor('xkcd:white') #Set the background = ‘white'
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    data2plot, lons2use = add_cyclic_point(data, coord=data.longitude)
    
    plot = ax.contourf(lons2use, data.latitude, data2plot, transform = ccrs.PlateCarree(), cmap = color_map, levels = levels, extend='both')
    ax.coastlines(resolution='110m'); gl=ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}; gl.ylabel_style = {'color': 'green', 'weight': 'bold'}
    ax.add_feature(cf.BORDERS)
    
    cax = fig.add_axes([0.17,0.1,0.7,0.03])
    cbar = fig.colorbar(plot, cax=cax, orientation='horizontal', shrink=1, aspect=12, ticks=levels)
    ax.set_title(title, fontsize=20)
    cbar.set_label(cbar_label, fontsize=16)
    plt.savefig('./%s.png' %('labs4dynamics_plots/'+filename),dpi=98)

    
### plotScalarAndWinds function creates a figure that contour-fills trends in input data and overlays trends in winds overlaid
def plotScalarAndWinds(data, uwind, vwind, color_map, title, cbar_label, filename, levels=None):
    fig = plt.figure(figsize=(14,18))
    fig.patch.set_facecolor('xkcd:white') #Set the background = ‘white'
    ax = plt.axes(projection=ccrs.PlateCarree())

    u = (uwind*12*100)*10
    v = (vwind*12*100)*10
    x = (data).longitude
    y = (data).latitude
    data2plot, lons2use = add_cyclic_point(data, coord=x)

    plot = ax.contourf(lons2use, y, data2plot, transform = ccrs.PlateCarree(), cmap = color_map, levels = levels, extend='both')

    ax.set_extent([-150, -70, 10, 70.50], ccrs.PlateCarree())
    ax.coastlines(resolution='110m'); gl=ax.gridlines(); gl.bottom_labels=True; gl.left_labels=True
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}; gl.ylabel_style = {'color': 'green', 'weight': 'bold'}
    ax.add_feature(cf.BORDERS); ax.add_feature(cf.STATES, linewidth=2)
    ax.barbs(x[::10],y[::10], u[::10,::10], v[::10,::10], sizes=dict(emptybarb=0),pivot='middle')   ### Overlay wind barbs

    cax = fig.add_axes([0.17,0.20,0.7,0.02])
    cbar = fig.colorbar(plot, cax=cax, orientation='horizontal', ticks = levels)
    ax.set_title(title, fontsize=20)
    cbar.set_label(cbar_label, fontsize=16)
    plt.savefig('./%s.png' %('labs4dynamics_plots/'+filename),dpi=98)

#Here's a function to calculate the slopes from linear regression using scipy.stats
from scipy import stats
linregress = lambda data: stats.linregress(np.arange(data.size), data).slope



                       ################################################################################
                                ### FUNCTIONS FOR STATISTICAL (MATHEMATICAL) ANALYSES ###
                       ################################################################################
            
#Here's a function to calculate the linear regression using xarray.Dataset.polyfit
def linregression_xr(data):
    linRegression = data.polyfit(dim='time', deg=1) #Function returns the coefficients of the best fit for each variable in this dataset.
    return linRegression


def trends(array):
    #Input array is f(time,latitude,longitude) 
    #Applies _linregress at each spatial grid point 
     
    trend = xr.apply_ufunc( 
        linregress, 
        array, 
        input_core_dims=[["time"]],  # list with one entry per arg 
        exclude_dims=set((["time"])),  # dimensions allowed to change size. Must be a set!
        dask ='allowed',
        vectorize=True,  # loop over non-core dims
    )   
    return ( trend )


#create a function to calculate wind speed of both u and v components of the wind
def wind_uv_speed(u,v):
    WSpd = (np.sqrt(np.square(u)+np.square(v)))
    return WSpd

