# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:23:26 2018

@author: ben.candy
"""
import requests
from pandas.io.json import json_normalize

def getward(lon, lat):
    url = "https://api.postcodes.io/postcodes?"
    url += "lon=%s&lat=%s&limit=1" % (lon, lat)
    resp = requests.get(url)
    j = resp.json()
    norm_data = json_normalize(j['result'])
    ward_data = norm_data.loc[:,['admin_ward','codes.admin_ward']]
    return(ward_data)
