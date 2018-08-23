import pandas as pd
from pandas.io.json import json_normalize
import time
from pandas import DataFrame
import requests

df_Foursquare = pd.DataFrame()

for i in range(1,50):
    for j in range(15,750,15):
        client_id = "XO1QQIQ02JE0EKQXZ4HBOYPHW2DSWJH5EIBN2AOZG2NRZVPK"
        client_secret = "GJGSIAZR4WXLYOHW4VP1JHTAMLH23EZXCWZDVPCZTYY2RQ4V"
        lat = (51.25 + (i/100))
        long = (-0.5 + (i/1000))
        category_id = '4d4b7104d754a06370d81259' # cultural space
        distance = 500
        requested_keys = ["categories","id","location","name"]
        url = "https://api.foursquare.com/v2/venues/search?ll=%s,%s&intent=browse&radius=%s&categoryId=%s&limit=2&client_id=%s&client_secret=%s&v=%s" % (lat, long, distance, category_id, client_id, client_secret, time.strftime("%Y%m%d"))
        resp = requests.get(url)
        dataResp = resp.json()
        if dataResp["response"]['venues'] != []: 
            data = DataFrame(dataResp["response"]['venues'])[requested_keys]
            df_FoursquareIteration = pd.DataFrame(data)
            df_FoursquareIteration["categories"] = df_Foursquare["categories"].apply(lambda x: dict(x[0])['name'])
            df_FoursquareIteration["lat"] = df_Foursquare["location"].apply(lambda x: dict(x)["lat"])
            df_FoursquareIteration["long"] = df_Foursquare["location"].apply(lambda x: dict(x)["lng"])
            df_Foursquare = pd.concat([df_Foursquare,df_FoursquareIteration])

df_Foursquare.drop_duplicates(subset=['id'], keep=False)

def point_in_poly(x,y,poly):
   # check if point is a vertex
   if (x,y) in poly: return "IN"
   # check if point is on a boundary
   for i in range(len(poly)):
      p1 = None
      p2 = None
      if i==0:
         p1 = poly[0]
         p2 = poly[1]
      else:
         p1 = poly[i-1]
         p2 = poly[i]
      if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
         return "IN"
      
   n = len(poly)
   inside = False

   p1x,p1y = poly[0]
   for i in range(n+1):
      p2x,p2y = poly[i % n]
      if y > min(p1y,p2y):
         if y <= max(p1y,p2y):
            if x <= max(p1x,p2x):
               if p1y != p2y:
                  xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
               if p1x == p2x or x <= xints:
                  inside = not inside
      p1x,p1y = p2x,p2y

   if inside: return "IN"
   else: return "OUT"
