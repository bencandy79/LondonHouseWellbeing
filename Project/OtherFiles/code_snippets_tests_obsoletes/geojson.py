# -*- coding: utf-8 -*-
"""
Created on Thu Sep 06 15:33:16 2018

@author: Ben Candy
"""

import pandas as pd 
import numpy as np
from pandas import json

 # create the feature collection
 
collection = {'type':'FeatureCollection', 'features':[]}
 
# function to create a feature from each row and add it to the collection
 
def feature_from_row(title, latitude, longitude, description):
   feature = { 'type': 'Feature',
              'properties': { 'title': '', 'description': ''},
              'geometry': { 'type': 'Point', 'coordinates': []}
              }
   feature['geometry']['coordinates'] = [longitude, latitude]
   feature['properties']['title'] = title
   feature['properties']['description'] = description
   collection['features'].append(feature) 
   return feature

# apply the feature_from_row function to populate the feature collection 
geojson_series = geojson_df.apply(lambda x: feature_from_row(x['title'],x['latitude'],x['longitude'],x['description']),axis=1)

 