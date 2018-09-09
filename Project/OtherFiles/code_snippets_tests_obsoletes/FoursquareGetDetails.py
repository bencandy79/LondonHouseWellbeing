# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:07:55 2018
@author: ben.candy
"""
import pandas as pd
from pandas.io.json import json_normalize
import time
from pandas import DataFrame
import requests
from Postcodes_API import getward

cats = {'food':"4d4b7105d754a06374d81259",'nightlife':"4d4b7105d754a06376d81259",'culture':"4d4b7104d754a06370d81259"}

def Get4SqCategoryDetails(category):
    client_id = "XO1QQIQ02JE0EKQXZ4HBOYPHW2DSWJH5EIBN2AOZG2NRZVPK"
    client_secret = "GJGSIAZR4WXLYOHW4VP1JHTAMLH23EZXCWZDVPCZTYY2RQ4V"
    lat = 51.51098570912317
    long = -0.1042606316131014
    distance = 20
    category_id = category

    df_Foursquare = DataFrame()

    requested_keys = ["categories","id","location","name"]

    url = "https://api.foursquare.com/v2/venues/search?ll=%s,%s&intent=browse&radius=%s&categoryId=%s&client_id=%s&client_secret=%s&v=%s" % (lat, long, distance, category_id, client_id, client_secret, time.strftime("%Y%m%d"))
    try:
        resp = requests.get(url)
        dataResp = resp.json()
        data = DataFrame(dataResp["response"]['venues'])[requested_keys]
        #print(data)
        df_Foursquare2 = DataFrame()
        venue_ids = []
        frames = []

        for d in data["id"]:                

            requested_keys2 = ["id", "rating", "likes.count"]

            url2 = "https://api.foursquare.com/v2/venues/%s?client_id=%s&client_secret=%s&v=%s" % (d, client_id, client_secret, time.strftime("%Y%m%d"))
            resp2 = requests.get(url2)
            dataResp2 = resp2.json()
            ddata = (dataResp2)['response']
            norm_data = json_normalize(ddata['venue'])

            if "rating" not in norm_data.columns:
                norm_data["rating"] = 0.0
         
            venue_ids.append(d)
            frames.append(norm_data[requested_keys2])
            time.sleep(1)

            df_Foursquare2 = pd.concat(frames, keys=venue_ids)
            df_Foursquare = pd.merge(data, df_Foursquare2, how='left',on='id')

        
        time.sleep(1) # stay within API limits

    except (Exception):
        print ("Exception")

    df_Foursquare["categories"] = df_Foursquare["categories"].apply(lambda x: dict(x[0])['name'])
    df_Foursquare["lat"] = df_Foursquare["location"].apply(lambda x: dict(x)["lat"])
    df_Foursquare["long"] = df_Foursquare["location"].apply(lambda x: dict(x)["lng"])
    df_Foursquare["distance"] = df_Foursquare["location"].apply(lambda x: dict(x)["distance"])
    df_Foursquare = df_Foursquare[["id","name","categories", "distance","lat","long","rating", "likes.count"]]

    df_ward = pd.DataFrame()

    add_ward = []

    for i in range(0,df_Foursquare.shape[0]):
        add_ward.append(getward(df_Foursquare.iat[i,4],df_Foursquare.iat[i,5]))

    df_ward = pd.concat(add_ward)
    
    df_FoursquareLocated = pd.merge(df_Foursquare, df_ward, how='left', on = 'id')

    return(df_FoursquareLocated)

Result = Get4SqCategoryDetails("4d4b7105d754a06374d81259")
