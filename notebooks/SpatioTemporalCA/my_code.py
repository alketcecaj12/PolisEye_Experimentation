import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import datetime
import geopandas as gpd
import folium
import time


locations_2 = []
loc_in = []
loc = []

def getgrid(loc, loc_in, locations_2, geo_df):
    for i in geo_df.geometry:
        stringa = str(i)
        
        coord = stringa[10:]
        coord = coord[:-2]
        co1 = coord.split(',')
        for j in co1:
            j = j.strip()
            co_0 = j.split(' ')
            loc_in.append(float(co_0[1].strip()))
            loc_in.append(float(co_0[0].strip()))
            loc.append(loc_in)
            loc_in = []
        locations_2.append(loc)
        loc = []
    return locations_2


def p(x):
    return x*2

def p_list(x):
    for i in range(len(x)):
        print(x[i])
    return x*2
