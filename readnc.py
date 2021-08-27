import xarray as xr
import numpy as np
from pydap.client import open_url
from pydap.cas.urs import setup_session
import regionmask
from benfords_law import BenfordsLaw
import os
username = 'dahlia_dry'
password = 'MassTech2023'

nasa_urllist = []
for jd in np.arange(1,366):
    url="https://oceandata.sci.gsfc.nasa.gov:443/opendap/MODISA/L3SMI/2003/" + '{:03d}'.format(jd) + "/A2003" + '{:03d}'.format(jd)+ ".L3m_DAY_CHL_chlor_a_9km.nc"
    #print(url)
    nasa_urllist.append(url)


def nasa_plot_benford(restrict_basin=False):
    """try:
        session = setup_session(username,password,check_url=url)
        pydap_ds = open_url(url, session=session)
        store = xr.backends.PydapDataStore(pydap_ds)
        ds=xr.open_dataset(store).assign_coords({"time": jd})
        ds = ds.chunk({'lon': 10})
        print('dataset successfully loaded')
    except Exception as err:
        print(jd)
        print(err)"""
    ds = xr.open_dataset("data/test/modis-chla-8d-2003.nc")
    #ds = xr.open_dataset("data/test/chlora_2013_361.nc4")
    print(ds)
    if restrict_basin:
        atlantic = [2.0,6.0]
        pacific = [3.0,4.0]
        basins = regionmask.defined_regions.natural_earth.ocean_basins_50
        print(regionmask.defined_regions.natural_earth.ocean_basins_50.map_keys('South Pacific Ocean'))
        mask = basins.mask(ds)
        data1= ds.where(mask ==4)
        data2 = ds.where(mask ==3)
        ds = xr.concat([data1,data2],dim='time')
    ds=ds.chunk({'lat':270,'lon':270,'time':23})
    #print(len(ds.chlor_a.values))
    #print(len(ds.chlor_a.values.flatten()))
    benfords = BenfordsLaw(ds.chlor_a.values.flatten())
    benfords.apply_benfords_law()
#copernicus
"""testfp = 'data/dataset-oc-glo-bio-multi-l3-chl_4km_daily-rep_1628190214403.nc'
print(os.listdir('/Volumes/dataset-oc-glo-bio-multi-l3-chl_4km_daily-rep/'))
testdata = xr.open_dataset(testfp)
print(testdata.dims)"""
#nasa_plot_benford("https://oceandata.sci.gsfc.nasa.gov:443/opendap/MODISA/L3SMI/2003/001/A2003001.L3m_DAY_CHL_chlor_a_9km.nc")
nasa_plot_benford()
