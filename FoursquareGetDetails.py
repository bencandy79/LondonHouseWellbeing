"""
Created on Fri Jul  6 11:07:55 2018

@author: ben.candy
"""

import pandas as pd
from pandas.io.json import json_normalize
import time
from pandas import DataFrame
import requests

client_id = "XO1QQIQ02JE0EKQXZ4HBOYPHW2DSWJH5EIBN2AOZG2NRZVPK"
client_secret = "GJGSIAZR4WXLYOHW4VP1JHTAMLH23EZXCWZDVPCZTYY2RQ4V"

lat = 51.51098570912317
long = -0.1042606316131014
distance = 100



df = DataFrame()
requested_keys = ["categories","id","location","name"]

url = "https://api.foursquare.com/v2/venues/search?ll=%s,%s&intent=browse&radius=%s&client_id=%s&client_secret=%s&v=%s" % (lat, long, distance, client_id, client_secret, time.strftime("%Y%m%d"))
try:
    resp = requests.get(url)
    dataResp = resp.json()
    data = DataFrame(dataResp["response"]['venues'])[requested_keys]
    

    df2 = DataFrame()
    venue_ids = []
    frames = []

    for d in data["id"]:                
        requested_keys2 = ["id", "rating", "likes.count"]

        url2 = "https://api.foursquare.com/v2/venues/%s?client_id=%s&client_secret=%s&v=%s" % (d, client_id, client_secret, time.strftime("%Y%m%d"))
        resp2 = requests.get(url2)
        dataResp2 = resp2.json()
        ddata = dataResp2['response']               

        nom_data = json_normalize(ddata['venue'])
        
        if "rating" not in nom_data.columns:
            nom_data["rating"] = 'NONE'                 

        venue_ids.append(d)
        frames.append(nom_data[requested_keys2])
        time.sleep(1)


        df2 = pd.concat(frames, keys=venue_ids)

        mdata = pd.merge(data, df2,how='left',on='id', suffixes=('_x', '_y'))

        df = df.append(mdata,ignore_index=True)
        print (df)
 
        time.sleep(1) # stay within API limits
except (Exception):
    print ("Exception")

df = df.drop_duplicates(subset='id',keep='last')
print (df)

df["categories"] = df["categories"].apply(lambda x: dict(x[0])['name'])
df["lat"] = df["location"].apply(lambda x: dict(x)["lat"])
df["long"] = df["location"].apply(lambda x: dict(x)["lng"])
df["distance"] = df["location"].apply(lambda x: dict(x)["distance"])
#df["checkins"] = df["stats"].apply(lambda x: dict(x)["checkinsCount"])

ordered_df = df[["id_x","name_x","categories", "distance","lat","long", "rating", "likes.count"]]

ordered_df.to_csv("C:/MSc Data Science/Project/Data/foursquare_%s_London.csv",encoding='utf-8', index=False)
