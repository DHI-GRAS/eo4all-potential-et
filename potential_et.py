# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:11:33 2023

@author: rmgu
"""

from dateutil import rrule
from pathlib import Path
import datetime as dt
import re
import math
import calendar

import numpy as np
import pysftp
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
import click

from pyTSEB.energy_combination_ET import penman_monteith
import pyTSEB.net_radiation as rad
import pyTSEB.resistances as res
import pyTSEB.meteo_utils as met

import interface


lc_parameters = ["veg_height", "leaf_type", "veg_fractional_cover", "min_stomatal_resistance",
                 "veg_inclination_distribution", "width_to_height_ratio",
                 "igbp_classification"]

meteo_variables = ["2m-temperature", "2m-dewpoint-temperature", "surface-pressure",
                   "surface-solar-radiation-downward", "10m-u-component-of-wind",
                   "10m-v-component-of-wind"]

scale_factor = 10000
output_scale_factor = 100
no_data = -9999


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


def daily_meteo_params(meteo_path, variables, start_date, end_date, template_file):

    meteo_params = {}
    wind = {}
    # Extract meteorological variables and convert to right units
    for variable in variables:
        variable_file = list(meteo_path.glob(f"*{variable}*_{start_date:%Y}_*"))[0]
        var_ds = xr.open_dataset(variable_file)
        var_ds.rio.write_crs("epsg:4326", inplace=True)
        var_data = var_ds.sel(time=slice(start_date, end_date))
        if variable == "2m-temperature":
            t_max = var_data.t2m.max("time")
            t_min = var_data.t2m.min("time")
            var_data = 0.5 * (t_max + t_min)
        elif variable == "2m-dewpoint-temperature":
            t_max = var_data.d2m.max("time")
            t_min = var_data.d2m.min("time")
            var_data = 0.5 * (t_max + t_min)
            var_data = calc_vapour_pressure(var_data)
            variable = "vapour-pressure"
        elif variable == "surface-pressure":
            # Convert pressure from pascals to mb
            var_data = var_data.sp.mean("time")/100.0
        elif variable == "surface-solar-radiation-downward":
            # Convert to average W m^-2
            var_data = var_data.ssrd.sum("time") / (end_date - start_date).total_seconds()
        elif variable == "10m-u-component-of-wind" or variable == "10m-v-component-of-wind":
            try:
                wind[variable] = var_data.u10
            except:
                wind[variable] = var_data.v10
            # Calculate absolute wind value
            if len(wind.keys()) == 2:
                var_data = (list(wind.values())[0]**2 +
                            list(wind.values())[1]**2)**0.5
                var_data = var_data.mean("time")
                variable = "10m-wind"
            else:
                continue

        # Reproject to the grid of other inputs
        with rasterio.open(template_file) as template:
            data = np.zeros(template.read(1).shape)
            reproject(var_data,
                      data,
                      src_transform=var_ds.rio.transform(),
                      src_crs=var_ds.rio.crs,
                      dst_transform=template.transform,
                      dst_crs=template.crs,
                      resampling=Resampling.bilinear)

        var_ds.close()

        meteo_params[variable] = data.astype(np.float32)

    return meteo_params


def download_data(connection, path, glob, overwrite=False):
    downloaded_files = []
    files = connection.listdir(str(path))
    for file in files:
        if re.search(glob, file):
            if overwrite or not Path(file).exists():
                connection.get(str(path / file))
            downloaded_files.append(Path(file))
    return downloaded_files


def upload_data(connection, path, file):
    try:
        connection.put(file, f"{path}/{file}")
    except IOError:
        connection.mkdir(path)  # Create remote_path
        connection.put(file, f"{path}/{file}")

def connect_to_server(ftp_url, ftp_port, ftp_username, ftp_pass):
    try:
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        connection = pysftp.Connection(host=ftp_url,
                                       port=ftp_port,
                                       username=ftp_username,
                                       password=ftp_pass,
                                       cnopts=cnopts)
        return connection
    except pysftp.ConnectionException:
        print("Could not connect to data source")
        return


def main(aoi_name, date, ftp_url, ftp_port, ftp_username, ftp_pass, spatial_res="s2",
         temporal_res="dekadal"):

    # Connect to sftp server
    connection = connect_to_server(ftp_url, ftp_port, ftp_username, ftp_pass)
    if connection is None:
        print("Cannot connect to server!")
        return

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

    # Download VI data
    print("Downloading VI data...")
    path = Path(f"/EO4ALL/{aoi_name}/{date_start:%Y%m%d}/10m/Vegetation-Indices")
    ndvi_file = download_data(connection, path, "_NDVI_")[0]
    evi_file = download_data(connection, path, "_EVI_")[0]
    refl_b2_file = download_data(connection, path, "_B02_")[0]
    refl_b4_file = download_data(connection, path, "_B04_")[0]
    refl_b8_file = download_data(connection, path, "_B08_")[0]
    refl_b11_file = download_data(connection, path, "_B11_")[0]
    refl_b12_file = download_data(connection, path, "_B12_")[0]

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
    landcover_file = download_data(connection, Path("/EO4ALL/Ancillaries/worldcover_data"), "_WorldCover_")[0]
    lut_file = Path("crop_coefficients_lut.csv")
    lc_maps = create_landcover_based_maps(landcover_file, lut_file, lc_parameters, ndvi_file, lai)

    print("Downloading meteorological data...")
    # Download meteorological data
    path = Path(f"/EO4ALL/Indonesia/{date_start:%Y}/30km/Climate-Indices/")
    era5_file = download_data(connection, path, f"_{date_start:%Y}_")[0]

    count = 0
    ta = np.zeros(lai.shape)
    wind = np.zeros(lai.shape)
    vapour = np.zeros(lai.shape)
    pressure = np.zeros(lai.shape)
    irradiance = np.zeros(lai.shape)
    print("Calculating daily meteo values...")
    for date in rrule.rrule(rrule.DAILY, dtstart=date_start, until=date_end):
        print(date)
        count = count + 1

        # Get daily meteo params
        # Most of Indonesia is in UTC+7 time zone
        start_date = dt.datetime(date.year, date.month, date.day, 0, 0, 0)
        start_date = start_date - dt.timedelta(hours=7)
        end_date = start_date + dt.timedelta(hours=24)
        meteo_maps = daily_meteo_params(Path(era5_file).parent,
                                        meteo_variables,
                                        start_date,
                                        end_date,
                                        ndvi_file)
        ta = ta + meteo_maps["2m-temperature"]
        wind = wind + meteo_maps["10m-wind"]
        vapour= vapour + meteo_maps["vapour-pressure"]
        pressure = pressure + meteo_maps["surface-pressure"]
        irradiance = irradiance + meteo_maps["surface-solar-radiation-downward"]

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

    out_file = f"EO4ALL_Potential-Evapotranspiration_ETp_S2-10m_{aoi_name}_{date_start:%Y%m%d}_{date_end:%Y%m%d}_{dt.datetime.now():%Y%m%d%H%M%S}.tif"
    with rasterio.open(ndvi_file, "r") as template:
        meta = template.meta
        meta.update({"driver": "COG"})
        with rasterio.open(out_file, "w", **meta) as fp:
            fp.scales = [1/output_scale_factor]
            et_p = et_p * output_scale_factor
            et_p[~valid_pixels] = no_data
            fp.write(et_p, 1)

    # Upload to FTP
    print("Uploading to FTP...")
    remote_path = Path(f"/EO4ALL/{aoi_name}/{date_start:%Y%m%d}/10m/Potential-Evapotranspiration")
    connection = connect_to_server(ftp_url, ftp_port, ftp_username, ftp_pass)
    if connection is None:
        print("Cannot connect to server!")
        return
    upload_data(connection, str(remote_path), out_file)

    # remove downloaded and produced files
    Path(ndvi_file).unlink()
    Path(evi_file).unlink()
    Path(refl_b2_file).unlink()
    Path(refl_b4_file).unlink()
    Path(refl_b8_file).unlink()
    Path(refl_b11_file).unlink()
    Path(refl_b12_file).unlink()
    Path(out_file).unlink()

    return str(remote_path)


@click.command()
@click.argument("json_data")
def run(json_data):
    inputs = interface.Inputs().loads(json_data)
    aoi_name = inputs["aoi_name"]
    date = inputs["date"]
    spatial_res = inputs["spatial_res"]
    temporal_res = inputs["temporal_res"]
    ftp_url = inputs["ftp_url"]
    ftp_port = inputs["ftp_port"]
    ftp_username = inputs["ftp_username"]
    ftp_pass = inputs["ftp_pass"]

    output_path = main(aoi_name, date, ftp_url, ftp_port, ftp_username, ftp_pass, spatial_res="s2",
                       temporal_res="dekadal")

    out = interface.Outputs().dumps({"output_path": output_path})
    return out


if __name__ == "__main__":
    run(standalone_mode=False)
