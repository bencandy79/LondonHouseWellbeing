"""
Created on Mon Jul 30 17:20:13 2018

@author: ben.candy
"""

import requests
from pandas.io.json import json_normalize
from Postcodes_API import getward
import pandas as pd

def getplace(lat, lon, place_type):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
    url += "location=%s,%s&radius=50&key=AIzaSyBrDHY2ZvRcV0Ob7yZbwEqLToTs3gALN0o" % (lat, lon)
    resp = requests.get(url)
    j = resp.json()
    norm_data = json_normalize(j['results'])
    norm_data["types"] = norm_data["types"].apply(lambda x: list(x)[0])
    data = norm_data[norm_data['types']== place_type]
    return(data)

data = (getplace(51.541694, -0.102172, 'point_of_interest'))

ward = pd.DataFrame(getward(data.iat[0,1],data.iat[0,0]))

for i in range(1,data.shape[0]):
    add_ward = getward(data.iat[i,1],data.iat[i,0])
    ward.loc[i] = add_ward
