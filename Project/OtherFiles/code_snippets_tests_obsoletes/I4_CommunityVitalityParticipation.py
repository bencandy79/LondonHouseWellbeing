# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:50:01 2018

@author: ben.candy
"""
import pandas as pd
from FoursquareGetDetails.py import Get4SqCategoryDetails


FOURSQ_FOOD_ID = "4d4b7105d754a06374d81259"
FOURSQ_NIGHTLIFE_ID = "4d4b7105d754a06376d81259"
FOURSQ_CULTURE_ID = "4d4b7104d754a06370d81259"
XLSX_2016_ELECTION = 'https://files.datapress.com/london/dataset/london-elections-results-2016-wards-boroughs-constituency/2016-05-27T10:46:12/gla-elections-votes-all-2016.xlsx'

# import election turnout data
try:
    df_election = pd.read_excel(XLSX_2016_ELECTION, sheet_name=1, skiprows=2)
except:
    print('Cannot open data file - Election 2016 by Ward')



# import Foursquare data
category_id = {'food':FOURSQ_FOOD_ID,'nightlife':FOURSQ_NIGHTLIFE_ID,'culture':FOURSQ_CULTURE_ID}

df_4sq_Food = Get4SqCategoryDetails(category_id['food'])
df_4sq_Nightlife = Get4SqCategoryDetails(category_id['nightlife'])
df_4sq_Culture = Get4SqCategoryDetails(category_id['culture'])

# combine Food and Nightlife dataframes to get one dataset for community vitality
df_4sq_BarsRestaurants = df_4sq_Food.append(df_4sq_Nightlife, ignore_index=True)
df_4sq_BarsRestaurants = df_4sq_BarsRestaurants.drop_duplicates(subset='id',keep='last')

# import election data
try:
    df_election = pd.read_excel(XLSX_2016_ELECTION, sheet_name=1, skiprows=2)
except:
    print('Cannot open data file - Election 2016 by Ward')
