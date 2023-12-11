# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:11:33 2023

@author: rmgu

Copyright (C) 2023  DHI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from dateutil import rrule
from glob import glob
from pathlib import Path
import datetime as dt
import click
import math
import calendar
import logging
import sys
import os

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from shapely.geometry import mapping
import gc

from pyTSEB.energy_combination_ET import penman_monteith
import pyTSEB.net_radiation as rad
import pyTSEB.resistances as res
import pyTSEB.meteo_utils as met

import interface


lc_parameters = ["veg_height", "leaf_type", "veg_fractional_cover", "min_stomatal_resistance",
                 "veg_inclination_distribution", "width_to_height_ratio",
                 "igbp_classification"]

meteo_variables = ["tmax", "dmax", "spavg", "ssrd", "wsavg"]

scale_factor = 10000
output_scale_factor = 100
no_data = -9999


def clipping2aoi(cog_files, df_S2_tile, df_S2_buffer_tile, epsg_info):

    # nodata in ESA Worldcover
    nodata = 0

    if len(cog_files) > 1:

        path2mosaic = []
        for path in cog_files:

            # Open ESA COG file
            xds = rxr.open_rasterio(path)

            # Reproject S2 buffered from GCS to ESA Worldcover EPSG
            df_S2_buffer_tile_rp = df_S2_buffer_tile.to_crs(xds.rio.crs)

            # Clip to aoi
            xds = xds.rio.clip(df_S2_buffer_tile_rp.geometry.apply(mapping))

            # Add to list
            path2mosaic.append(xds)

            # Close and delete it
            xds.close()
            del xds

        # Merge/Mosaic multiple rasters using merge_arrays method of rioxarray
        merged_xds = merge_arrays(dataarrays=path2mosaic,
                                     res=(10, 10), crs=epsg_info,
                                     nodata=nodata)

    else:

        # Open ESA COG file
        xds = rxr.open_rasterio(cog_files[0])

        # Reproject S2 buffered from GCS to ESA Worldcover EPSG
        df_S2_buffer_tile_rp = df_S2_buffer_tile.to_crs(xds.rio.crs)

        # Clip to aoi
        xds = xds.rio.clip(df_S2_buffer_tile_rp.geometry.apply(mapping))

        # Merge/Mosaic multiple rasters using merge_arrays method of rioxarray
        merged_xds = merge_arrays(dataarrays=xds,
                                     res=(10, 10), crs=epsg_info,
                                     nodata=nodata)

        # Close and delete it
        xds.close()
        del xds

    # Last step: Clip merged raster to S2 tile without buffer

    # Reproject S2 buffered from GCS to S2 UTM huse
    df_S2_tile_rp = df_S2_tile.to_crs(merged_xds.rio.crs)

    # Clip to aoi
    merged_xds = merged_xds.rio.clip(df_S2_tile_rp.geometry.apply(mapping))

    return merged_xds


def esa_worldcover(tile):

    # ancillaries root directory
    ancillaries_rootpath = r'/data/_ancillaries'

    # output folder
    output_dir = r'/data/testFolder/ESA_Worldcover'

    # output filepath
    filepath_out = os.path.join(output_dir, 'esa_worldcover_2021_' + tile + '.tif')
    if os.path.exists(filepath_out):
        return filepath_out

    # S2 worldtiles
    S2_worldtiles_shp = os.path.join(ancillaries_rootpath,
                                     'S2_worldtiles',
                                     'S2_worldtiles.shp')

    # S2 worldtiles buffer 500 meters
    S2_worldtiles_bu500m_geojson = os.path.join(ancillaries_rootpath,
                                            'S2_worldtiles',
                                            'S2_worldtiles_buffer-500m.geojson')

    # ESA Worldcover 2021 tiles
    esa_worldcover_shp = os.path.join(ancillaries_rootpath,
                                      'ESA-Worldcover_worldtiles',
                                      'ESA_WorldCover_10m_2021_v200_60deg.shp')

    # S2_worldtiles
    df_S2 = gpd.read_file(S2_worldtiles_shp)

    # S2_worldtiles buffer 500m
    df_S2_buffer = gpd.read_file(S2_worldtiles_bu500m_geojson, driver='GeoJSON')

    # ESA Worldcover 2021
    df_ESA = gpd.read_file(esa_worldcover_shp)


    # S2_worldtiles
    df_S2_tile = df_S2[df_S2['Name'] == tile[1:]] # remove 'T'

    # S2_worldtiles buffer 500m
    df_S2_buffer_tile = df_S2_buffer[df_S2_buffer['TILE'] == tile]

    # epsg to reproject later
    epsg_info = df_S2_buffer_tile['EPSG_INFO'].values.tolist()[0]

    # Get the intersection of the two GeoDataFrames
    intersection = gpd.overlay(df_ESA, df_S2_tile, how='intersection')

    # Get the ESA Worldcover tiles
    cog_files = intersection['name'].values.tolist()

    # Add path to files
    for i in range(len(cog_files)):
        cog_files[i] = os.path.join(ancillaries_rootpath,
                                        'ESA-Worldcover_worldtiles',
                                        'COG_files', cog_files[i] + '.tif')

    # Do clipping to buffered S2 tile, GeoTIFF merging and clipping again to S2 tile without buffer
    merged_xds = clipping2aoi(cog_files, df_S2_tile, df_S2_buffer_tile, epsg_info)

    # Save in GeoTIFF format
    merged_xds.rio.to_raster(filepath_out)

    # Close and delete it
    merged_xds.close()
    del merged_xds

    # Freeing up memory and improving performance
    collected = gc.collect()

    return filepath_out


def calc_lai(evi):
    ''' Based on
    Eva Boegh, H. Soegaard, N. Broge, C.B. Hasager, N.O. Jensen, K. Schelde, A. Thomsen,
    Airborne multispectral data for quantifying leaf area index, nitrogen concentration, and photosynthetic efficiency in agriculture,
    Remote Sensing of Environment, Volume 81, Issues 2–3, 2002, Pages 179-193, ISSN 0034-4257,
    https://doi.org/10.1016/S0034-4257(01)00342-X.
    (https://www.sciencedirect.com/science/article/pii/S003442570100342X)
    '''
    lai = 3.618 * evi - 0.118
    return np.maximum(lai, 0)


def calc_emis(ndvi, red_refl=None):
    ''' Based on
    J.A. Sobrino, J.C. Jiménez-Muñoz, G. Sòria, A.B. Ruescas, O. Danne, C. Brockmann, D. Ghent, J. Remedios, P. North, C. Merchant, M. Berger, P.P. Mathieu, F.-M. Göttsche,
    Synergistic use of MERIS and AATSR as a proxy for estimating Land Surface Temperature from Sentinel-3 data,
    Remote Sensing of Environment, Volume 179, 2016, Pages 149-161, ISSN 0034-4257,
    https://doi.org/10.1016/j.rse.2016.03.035.
    (https://www.sciencedirect.com/science/article/pii/S0034425716301158)
    '''

    # Constant parameters
    ndvi_soil = 0.15
    ndvi_veg = 0.9
    emissivity_soil = 0.973  # Average of SLSTR B11 and B12 emissivities from Sec 4.1
    emissivity_veg = 0.99
    red_params = [-0.04, 0.981]  # Average of SLSTR B11 and B12 params from Table 2

    # First calculate fractional vegetation cover
    ndvi[ndvi < ndvi_soil] = ndvi_soil
    ndvi[ndvi > ndvi_veg] = ndvi_veg
    fractional_vegetation_cover = (ndvi - ndvi_soil)/(ndvi_veg - ndvi_soil)

    # Then calculate broadband emissivity
    emissivity = emissivity_soil * (1 - fractional_vegetation_cover) + \
                 emissivity_veg * fractional_vegetation_cover

    # Adjust bare soil emissvity if required
    if red_refl is not None and red_params:
        i = fractional_vegetation_cover <= 0.05
        emissivity[i] = red_params[0] * red_refl[i] + red_params[1]

    return emissivity


def calc_albedo(refl_b2, refl_b4, refl_b8, refl_b11, refl_b12):
    ''' Albedo calculation for S2 BOA reflectances using the method and parameters
        from Liang (2000), Narrowband to broadband conversions of land surface albedo
        I Algorithms. The parameters are for Landsat 7 (ETM+). Landsat 7 and S2 have
        a bit different spectral chanels but the method has been used previously
        with OK results (Naegeli 2017 - http://www.mdpi.com/2072-4292/9/2/110/htm)
    '''

    # Constants from eq 11
    s2 = 0.356
    s4 = 0.130
    s8 = 0.373
    s11 = 0.085
    s12 = 0.072
    s0 = -0.0018

    # Calculate shortwave albedo
    alpha_short = s2 * refl_b2 + s4 * refl_b4 + s8 * refl_b8 + s11 * refl_b11 + s12 * refl_b12 + s0

    return alpha_short


def calc_longwave_irrradiance(ea, ta, p, z_t=2):
    longwave_irradiance = rad.calc_longwave_irradiance(ea, ta, p, z_t)
    return longwave_irradiance


def calc_roughness(lai, h_C, w_C, landcover, f_c):
    z_0M, d_0 = res.calc_roughness(lai,
                                   h_C,
                                   w_C,
                                   landcover,
                                   f_c)
    return z_0M, d_0


def calculate_potential_et(lai, emis, albedo, f_c, h_C, x_LAD, w_C, rst_min, leaf_type, landcover,
                           ta, u, ea, p, shortwave_irradiance, z_t, z_u, valid_pixels):

    net_shortwave_radiation = shortwave_irradiance * (1 - albedo)

    longwave_irradiance = calc_longwave_irrradiance(ea, ta, p, z_t=z_t)

    z_0M, d_0 = calc_roughness(lai, h_C, w_C, landcover, f_c)

    et_p = np.zeros(ta.shape, np.float32)
    le = penman_monteith(ta[valid_pixels],
                         u[valid_pixels],
                         ea[valid_pixels],
                         p[valid_pixels],
                         net_shortwave_radiation[valid_pixels],
                         longwave_irradiance[valid_pixels],
                         emis[valid_pixels],
                         lai[valid_pixels],
                         z_0M[valid_pixels],
                         d_0[valid_pixels],
                         z_u,
                         z_t,
                         calcG_params=[[1], 0.35],
                         const_L=None,
                         Rst_min=rst_min[valid_pixels],
                         leaf_type=leaf_type[valid_pixels],
                         f_cd=None,
                         kB=2.3,
                         verbose=True)[3]
    et_p[valid_pixels] = met.flux_2_evaporation(le,
                                                ta[valid_pixels],
                                                time_domain=24)
    et_p = np.maximum(et_p, 0)
    return et_p


def create_landcover_based_maps(landcover_file, lut_file, parameters, template_file, lai):

    # Resample and subset the landcover parameter to output resolution and extent
    with rasterio.open(landcover_file) as lc, rasterio.open(template_file) as template:
        landcover = np.zeros(template.read(1).shape).astype(np.float32)
        reproject(lc.read(1),
                  landcover,
                  src_transform=lc.transform,
                  src_crs=lc.crs,
                  dst_transform=template.transform,
                  dst_crs=template.crs,
                  resampling=Resampling.nearest)

    # Read the landcover LUT
    with open(lut_file, 'r') as fp:
        lines = fp.readlines()
    headers = lines[0].rstrip().split(',')
    values = [x.rstrip().split(',') for x in lines[1:]]
    lut = {}
    for idx, key in enumerate(headers):
        try:
            lut[key] = [float(x[idx]) for x in values if len(x) == len(headers)]
        except ValueError:
            lut[key] = [x[idx] for x in values if len(x) == len(headers)]

    # Create arrays for the landcover dependent parameters
    data = {}
    for param in parameters:
        if param not in lut.keys():
            print("Parameter %s is not in the look-up-table %s! Skipping!" % (param, lut_file))
            continue
        temp = np.zeros(landcover.shape, dtype=np.float32) + np.NaN
        if param == "veg_height":
            lai_max = np.zeros(landcover.shape) + np.NaN
            min_height = np.zeros(landcover.shape) + np.NaN

        # Set the parameters for each present land cover class
        for lc_class in np.unique(landcover):
            if lc_class not in lut['class']:
                continue
            lc_pixels = np.where(landcover == lc_class)
            lc_index = lut['class'].index(lc_class)
            temp[lc_pixels] = lut[param][lc_index]
            # Update herbaceous canopy height for crops based on LAI
            if param == "veg_height":
                if lut["veg_height"][lc_index] != lut["min_veg_height"][lc_index]:
                    lai_max[lc_pixels] = lut["lai_max"][lc_index]
                    min_height[lc_pixels] = lut["min_veg_height"][lc_index]
                    temp[lc_pixels] = (temp[lc_pixels] *
                                       np.minimum((lai[lc_pixels]/lai_max[lc_pixels])**3.0, 1.0))
                    temp[lc_pixels] = np.maximum(min_height[lc_pixels], temp[lc_pixels])

        data[param] = temp

    return data


def calc_vapour_pressure(td):
    # Input - dew point temperature in K
    # Output - vapour pressure in mb
    td = td - 273.15
    e = 6.11 * np.power(10, (7.5 * td)/(237.3 + td))
    return e


def daily_meteo_params(path, variables, date, template_file):

    meteo_params = {}
    with rasterio.open(template_file) as template:
        template_transform = template.transform
        template_crs = template.crs
        template_shape = template.read(1).shape

    # Extract meteorological variables and convert to right units
    for variable in variables:
        var_path = get_files(path, f"*_{variable}_*_{date:%Y%m%d}_*")[0]
        var_data = rasterio.open(var_path).read(1)
        if variable == "tmax":
            # Need to also read tmin to calculate the average and convert to K
            t_min = rasterio.open(get_files(path, f"*_tmin_*_{date:%Y%m%d}_*")[0]).read(1)
            var_data = 0.5 * (var_data + t_min) + 273.15
            variable = "tavg"
        elif variable == "dmax":
            # Need to also read dmin to calculate the average and convert to vapour pressure
            d_min = rasterio.open(get_files(path, f"*_dmin_*_{date:%Y%m%d}_*")[0]).read(1)
            var_data = 0.5 * (var_data + d_min) + 273.15
            var_data = calc_vapour_pressure(var_data)
            variable = "vpavg"
        elif variable == "ssrd":
            # Need to convert to average daily W m^-2
            var_data = var_data / dt.timedelta(hours=24).total_seconds() * 1000000
        elif variable == "wsavg":
            # Set minimum windspeed to 1 m/s
            var_data = np.maximum(var_data, 1.0)

        # Reproject to the grid of other inputs
        with rasterio.open(var_path) as var:
            data = np.zeros(template_shape)
            reproject(var_data,
                      data,
                      src_transform=var.transform,
                      src_crs=var.crs,
                      dst_transform=template_transform,
                      dst_crs=template_crs,
                      resampling=Resampling.bilinear)

        meteo_params[variable] = data.astype(np.float32)

    return meteo_params


def get_files(path, glob):
    return list(path.glob(glob))


def main(aoi_name, date, spatial_res="s2", temporal_res="dekadal"):

    # Find start of dekade or month within which the date falls
    if temporal_res == "dekadal":
        date_start = dt.date(date.year, date.month, int(math.floor(date.day/10)*10 + 1))
        if date_start.day == 21:
            date_end = dt.date(date.year, date.month, calendar.monthrange(date.year, date.month)[1])
        else:
            date_end = date_start + dt.timedelta(days=9)
    elif temporal_res == "monthly":
        date_start = dt.date(date.year, date.month, 1)
        date_end = dt.date(date.year, date.month, calendar.monthrange(date.year, date.month)[1])
    else:
        date_start = date
        date_end = date

    print(f"Running for aoi {aoi_name} date start {date_start:%Y%m%d} and date end {date_end:%Y%m%d}")

    # Create output file name
    out_folder = Path(f"/data/outputs/{aoi_name}/{date_start:%Y%m%d}/10m/Potential-Evapotranspiration")
    out_file = out_folder / f"Potential-Evapotranspiration_ETp_S2-10m_{aoi_name}_{date_start:%Y%m%d}-{date_end:%Y%m%d}_{dt.datetime.now():%Y%m%d%H%M%S}.tif"
    out_file_existence_check = out_folder / f"Potential-Evapotranspiration_ETp_S2-10m_{aoi_name}_{date_start:%Y%m%d}-{date_end:%Y%m%d}_*.tif"
    existing_files = glob(str(out_file_existence_check))
    # Check if output file already exists and return if it does
    if os.getenv("DEBUG", None) is None and existing_files:
        print(f"File {str(existing_files[0])} exists. Skipping calculation...")
        return str(existing_files[0])

    print(f"File {str(out_file)} does not exist. Calculating...")

    # Download VI data
    print("Accessing VI data...")
    path = Path(f"/data/outputs/{aoi_name}/{date_start:%Y%m%d}/10m/Vegetation-Indices")
    ndvi_file = get_files(path, "*_NDVI_*")[0]
    evi_file = get_files(path, "*_EVI_*")[0]
    refl_b2_file = get_files(path, "*_B02_*")[0]
    refl_b4_file = get_files(path, "*_B04_*")[0]
    refl_b8_file = get_files(path, "*_B08_*")[0]
    refl_b11_file = get_files(path, "*_B11_*")[0]
    refl_b12_file = get_files(path, "*_B12_*")[0]

    print("Calculating biophysical parameters...")
    # Calculate LAI, emissivity and albedo
    with rasterio.open(evi_file) as evi:
        lai = calc_lai(evi.read(1).astype(np.float32) / scale_factor)
        valid_pixels = evi.read(1) != no_data
    with rasterio.open(ndvi_file) as ndvi, rasterio.open(refl_b4_file) as red:
        emis = calc_emis(ndvi.read(1).astype(np.float32) / scale_factor,
                         red.read(1).astype(np.float32) / scale_factor)
    with (rasterio.open(refl_b2_file) as refl_b2, rasterio.open(refl_b4_file) as refl_b4,
          rasterio.open(refl_b8_file) as refl_b8, rasterio.open(refl_b11_file) as refl_b11,
          rasterio.open(refl_b12_file) as refl_b12):
        albedo = calc_albedo(refl_b2.read(1).astype(np.float32) / scale_factor,
                             refl_b4.read(1).astype(np.float32) / scale_factor,
                             refl_b8.read(1).astype(np.float32) / scale_factor,
                             refl_b11.read(1).astype(np.float32) / scale_factor,
                             refl_b12.read(1).astype(np.float32) / scale_factor)

    print("Setting landcover parameters...")
    # Get parameters based on crop classification
    landcover_file = esa_worldcover(aoi_name)
    lut_file = Path("crop_coefficients_lut.csv")
    lc_maps = create_landcover_based_maps(landcover_file, lut_file, lc_parameters, ndvi_file, lai)

    print("Accessing daily meteorological data...")
    # Download meteorological data
    path = Path(f"/data/outputs/{aoi_name}/{date_start:%Y%m%d}/9km/Climate-Indices")
    count = 0
    ta = np.zeros(lai.shape)
    wind = np.zeros(lai.shape)
    vapour = np.zeros(lai.shape)
    pressure = np.zeros(lai.shape)
    irradiance = np.zeros(lai.shape)
    for date in rrule.rrule(rrule.DAILY, dtstart=date_start, until=date_end):
        print(date)
        count = count + 1

        # Get daily meteo params
        meteo_maps = daily_meteo_params(path,
                                        meteo_variables,
                                        date,
                                        ndvi_file)

        ta = ta + meteo_maps["tavg"]
        wind = wind + meteo_maps["wsavg"]
        vapour = vapour + meteo_maps["vpavg"]
        pressure = pressure + meteo_maps["spavg"]
        irradiance = irradiance + meteo_maps["ssrd"]

    ta = ta / count
    wind = wind / count
    vapour = vapour / count
    pressure = pressure / count
    irradiance = irradiance / count

    # Calculate potential ET
    valid_pixels = np.logical_and(valid_pixels,
                                  np.isfinite(lc_maps["veg_height"]))
    print("Calculating potential ET...")
    et_p = calculate_potential_et(lai,
                                  emis,
                                  albedo,
                                  lc_maps["veg_fractional_cover"],
                                  lc_maps["veg_height"],
                                  lc_maps["veg_inclination_distribution"],
                                  lc_maps["width_to_height_ratio"],
                                  lc_maps["min_stomatal_resistance"],
                                  lc_maps["leaf_type"],
                                  lc_maps["igbp_classification"],
                                  ta,
                                  wind,
                                  vapour,
                                  pressure,
                                  irradiance,
                                  2,
                                  10,
                                  valid_pixels)

    out_folder.mkdir(parents=True, exist_ok=True)
    print("Saving output file...")
    with rasterio.open(ndvi_file, "r") as template:
        meta = template.meta
        meta.update({"driver": "COG"})
        with rasterio.open(out_file, "w", **meta) as fp:
            fp.scales = [1/output_scale_factor]
            et_p = et_p * output_scale_factor
            et_p[~valid_pixels] = no_data
            fp.write(et_p, 1)

    return str(out_file)


def run(json_data):

    # Setup logging to file
    log_file = Path(f"/data/logs/docker_PR13_Potential-Evapotranspiration_logs/Potential_Evapotranspiration_{dt.datetime.now():%Y%m%d%H%M%S}.log")
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    sys.stderr.write = logger.error
    sys.stdout.write = logger.info

    inputs = interface.Inputs().loads(json_data)
    aoi_name = inputs["aoi_name"]
    date = inputs["date"]
    spatial_res = inputs["spatial_res"]
    temporal_res = inputs["temporal_res"]

    output_path = main(aoi_name, date, spatial_res=spatial_res, temporal_res=temporal_res)

    return output_path


@click.command()
@click.argument("json_data")
def cli_main(json_data):
    run(json_data)



if __name__ == "__main__":
    cli_main()
