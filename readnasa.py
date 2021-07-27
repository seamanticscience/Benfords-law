import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
import os
#import xarray as xr
import urllib.request
import requests
import wget

urls = [line.strip().replace('\n','') for line in open('data/chlorophyll_binned-2010-2021/urls.txt')]
def getdata():
    for url in urls:
        r = requests.get(url)
        with open('data/chlorophyll_binned-2010-2021/'+url.split('/')[-1], 'wb') as f:
            f.write(r.content)

def getdatav2():
    for url in urls:
        wget.download(url, 'data/chlorophyll_binned-2010-2021/'+url.split('/')[-1])
#datafiles = [x for x in os.listdir('data/chlorophyll_binned-2010-2021/') if x.split('.')[-1]=='nc']
#dataset = netCDF4.Dataset('data/chlorophyll_binned-2010-2021/'+datafiles[0])
#ds = xr.open_dataset('data/chlorophyll_binned-2010-2021/'+datafiles[0],engine='netcdf4')
#df = ds.to_dataframe()
#print(df.head())
#print(dataset)

getdatav2()
