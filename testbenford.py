#from benfordslaw import benfordslaw
from benfords_law import BenfordsLaw
from wodpy import wod
import pandas as pd
import geopandas as gpd
import numpy as np
import regionmask

#test
#fid = open("data/ocldb1624454431.12751.OSD") #2010-2021
#output_name = "data/test/testset.csv"

def testbenford1(data,col,z = None): #compile for one depth, use benfordslaw package
    data = data[data[col]>=0] #filter out blanks
    if z is None:
        z = data['z'].iloc[0]
    bl = benfordslaw(alpha=0.05)
    X = data[col].loc[data['z']==z].values
    print(len(X))
    results = bl.fit(X)
    bl.plot(title=col)
    # throws segmentation fault error

def testbenford1b(data,col,z = None): #compile for one depth, use benfords-law package
    data = data[data[col]>=0] #filter out blanks
    if z is None:
        z = data['z'].iloc[0]
    X = data[col].loc[data['z']==z].values
    benfords = BenfordsLaw(X,col)
    benfords.apply_benfords_law()


#testbenford1b(pd.read_csv('testset.csv'),'t')

def preprocess(fid,output_name):
    frames = []
    lastprofile= False
    while not lastprofile:
        try:
            profile = wod.WodProfile(fid)
            frame = profile.df().copy()
            frame["latitude"] = [profile.df().meta['latitude'] for x in range(len(frame))]
            frame["longitude"] = [profile.df().meta['longitude'] for x in range(len(frame))]
            frames.append(frame)
            print(len(frames))
            lastprofile = profile.is_last_profile_in_file(fid)
        except:
            break
    data = pd.concat(frames)
    data = data.fillna(-1)
    #print(data.head())
    data.to_csv(output_name)


class BenfordWOD(object):
    def __init__(self,data,var,control=None,control_val=None):
        """
        Parameters:
           -------------------------------------------------------------------
           data (str): filepath to WOD output file
           var (str): variable to test benfordness of (e.g. o2 concentration, temp)
           control (str): variable on which to partition the data (e.g. depth (z), basin)
           control_val: value of control used to partition the data (e.g. depth=200)
           output_name (str): filepath of output formatted data
           --------------------------------------------------------------------
        """
        self.data = data
        self.var = var
        self.control = control
        if control_val is None and control is not None:
            self.control_val=self.data[self.data[self.var]>=0][self.var].iloc[0]
        else:
            self.control_val = control_val

    def testbenford(self):
        testdata = self.data[self.data[self.var]>=0] #filter out blanks
        print(testdata)
        if self.control is not None and (type(self.control_val) is int or type(self.control_val) is float) :
            X = testdata[self.var].loc[testdata[self.control]==self.control_val].values
        elif self.control is not None and type(self.control_val) is tuple:
            X = testdata[self.var].loc[(testdata[self.control]>self.control_val[0]) & (testdata[self.control]<self.control_val[1])].values
        print(X)
        benfords = BenfordsLaw(X,self.var,self.control,self.control_val)
        benfords.apply_benfords_law()

def makecsvs():
    fids = [open("data/2010-2021-all/ocldb1625108116.22161.OSD"),
            open("data/2010-2021-all/ocldb1625108116.22161.CTD"),
            open("data/2010-2021-all/ocldb1625108116.22161.CTD2"),
            open("data/2010-2021-all/ocldb1625108116.22161.CTD3"),
            open("data/2010-2021-all/ocldb1625108116.22161.CTD4"),
            open("data/2010-2021-all/ocldb1625108116.22161.CTD5"),
            open("data/2010-2021-all/ocldb1625108116.22161.CTD6")
            ]
    f = ['osd','ctd','ctd2','ctd3','ctd4','ctd5','ctd6']
    for i in range(len(fids)):
        preprocess(fids[i],"data/2010-2021-all/" + f[i] + ".csv")

def runtests_depth_osd():
    osd = BenfordWOD(pd.read_csv('data/2010-2021-all/osd.csv'),'oxygen','z',(200,500))
    osd.testbenford()
    osd = BenfordWOD(pd.read_csv('data/2010-2021-all/osd.csv'),'oxygen','z',500)
    osd.testbenford()

def runtests_depth_ctd():
    ctd = BenfordWOD(pd.read_csv('data/2010-2021-all/ctd.csv'),'oxygen','z',(200,500))
    ctd.testbenford()
    ctd = BenfordWOD(pd.read_csv('data/2010-2021-all/ctd.csv'),'oxygen','z',500)
    ctd.testbenford()

def runtests_basin_osd():
    basins = regionmask.defined_regions.natural_earth.ocean_basins_50
    lon = np.arange(0.5, 360)
    lat = np.arange(89.5, -90, -1)
    mask = basins.mask(lon, lat)
    #print(mask[round(58.38),round(8.82)])
    print(np.where(basins.numbers==21))
    print(basins.names[np.where(basins.numbers==21)[0]])
    print(mask[50,50])
