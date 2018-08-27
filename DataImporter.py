# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:29:29 2018
@author: Ben Candy
"""
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import time
import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform

def importHousePrices():
    XLS_HOUSE_PRICE = 'https://data.london.gov.uk/download/average-house-prices/fb8116f5-06f8-42e0-aa6c-b0b1bd69cdba/land-registry-house-prices-ward.xls'
    try:
        df_HousePricesAll = pd.read_excel(XLS_HOUSE_PRICE,'Median',skiprows=[1])
    except:
        print('Cannot open data file - London House Prices')
    df_HousePrices = pd.DataFrame(df_HousePricesAll.iloc[1:,:])
    df_HousePrices.to_csv('files/HousePrices.csv',index=False)
    return(df_HousePrices)
    
def importEmploymentRates():
    XLSX_WARD_ATLAS = 'https://files.datapress.com/london/dataset/ward-profiles-and-atlas/2015-09-24T14:48:35/ward-atlas-data.xls'
    try:
        df_ward_atlas2 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet2', skiprows=2)   
    except:
        print('Cannot open data file - Ward Atlas Sheet 2') 
    employment_columns = ['New Code','Economically active: % In employment.1']
    df_employment = pd.DataFrame(df_ward_atlas2, columns = employment_columns)
    df_employment.rename(columns={'New Code':'Ward_Code','Economically active: % In employment.1':'Employ_Rate'},inplace=True)
    df_employment.to_csv('files/Employment.csv',index=False)
    return(df_employment)
    
def importGCSE():
    XLSX_WARD_ATLAS = 'https://files.datapress.com/london/dataset/ward-profiles-and-atlas/2015-09-24T14:48:35/ward-atlas-data.xls'
    try:
        df_ward_atlas3 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet3', skiprows=2)   
    except:
        print('Cannot open data file - Ward Atlas Sheet 3')
    GCSE_columns = ['New Code','2014.2']
    df_GCSE = pd.DataFrame(df_ward_atlas3, columns = GCSE_columns)
    df_GCSE.rename(columns={'New Code':'Ward_Code','2014.2':'Av_GCSE_Points'},inplace=True)
    df_GCSE.to_csv('files/GCSE.csv',index=False)
    return(df_GCSE)
    
def importSchools():
    CSV_SCHOOLS = 'http://ea-edubase-api-prod.azurewebsites.net/edubase/edubasealldata20180827.csv'
    try:
        iter_csv = pd.read_csv(CSV_SCHOOLS, iterator=True, chunksize=1000)
        df_schoolsall = pd.DataFrame()
        df_schoolsall = pd.concat([chunk[chunk['GOR (name)'] == 'London'] for chunk in iter_csv])
    except:
        print('Cannot open data file - SchoolsData')
    df_schoolslondonopen = df_schoolsall[(df_schoolsall['SchoolCapacity'] > 0) & (df_schoolsall['EstablishmentStatus (code)'] == 1)]
    df_schoolslondonopen['OfstedRating (name)'].replace('', np.nan, inplace=True)
    df_schoolslondonopen.dropna(subset=['OfstedRating (name)'], inplace=True)
    df_schoolslondon = pd.DataFrame(df_schoolslondonopen)
    df_schoolslondon.rename(columns={'OfstedRating (name)':'Ofsted_Rating'}, inplace=True)
    ofsted_ratings = {'Ofsted_Rating':{'Outstanding':2,'Good':1,'Requires improvement':-1,'Serious Weaknesses':-2,'Special Measures':-2,'Inadequate':-2}}
    df_schoolslondon.replace(ofsted_ratings,inplace=True)
    df_schoolslondon.to_csv('files/Schools.csv',index=False)
    return(df_schoolslondon)
    
def importCrime():
    CSV_CRIME = 'https://data.london.gov.uk/download/recorded_crime_summary/866c05de-c5cd-454b-8fe5-9e7c77ea2313/MPS%20Ward%20Level%20Crime%20%28most%20recent%2024%20months%29.csv'
    try:
        df_crime = pd.read_csv(CSV_CRIME)
    except:
        print('Cannot open data file - Crime by Ward')
    df_crime['2017'] = df_crime['201701']+df_crime['201702']+df_crime['201703']+df_crime['201704']+df_crime['201705']+df_crime['201706']+df_crime['201707']+df_crime['201708']+df_crime['201709']+df_crime['201710']+df_crime['201711']+df_crime['201712']
    df_crime.to_csv('files/Crime.csv', index=False)
    return(df_crime)
    
def importTraffic():
    XLSX_TRAFFIC_FATALITIES = 'https://files.datapress.com/london/dataset/road-casualties-severity-borough/road-casualties-severity-lsoa-msoa-ward.xls'
    try:
        df_traffic_f = pd.read_excel(XLSX_TRAFFIC_FATALITIES, 'Ward 2014 only', skiprows=1)
        df_traffic_f2 = pd.read_excel(XLSX_TRAFFIC_FATALITIES, 'Ward(pre2014)', skiprows=1)
    except:
        print('Cannot open data file - Traffic Fatalities by Ward')
    traffic_columns = ['Unnamed: 0','1 Fatal.9','2 Serious.9','3 Slight.9']
    traffic_columns2 = ['Ward Code','Fatal.4','Serious.4','Slight.4']
    df_traffic_fatalities = pd.DataFrame(df_traffic_f, columns = traffic_columns)
    df_traffic_fatalities2 = pd.DataFrame(df_traffic_f2, columns = traffic_columns2)
    df_traffic_fatalities.rename(columns={'Unnamed: 0':'Ward_Code','1 Fatal.9':'Fatal','2 Serious.9':'Serious','3 Slight.9':'Slight'}, inplace=True)
    df_traffic_fatalities2.rename(columns={'Ward Code':'Ward_Code','Fatal.4':'Fatal','Serious.4':'Serious','Slight.4':'Slight'}, inplace=True)
    df_traffic_fatalities = pd.concat([df_traffic_fatalities, df_traffic_fatalities2])
    df_traffic_fatalities.to_csv('files\Traffic.csv', index=False)

def importEmissions():
    XLSX_WARD_ATLAS = 'https://files.datapress.com/london/dataset/ward-profiles-and-atlas/2015-09-24T14:48:35/ward-atlas-data.xls'
    try:
        df_ward_atlas5 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet5', skiprows=2)   
    except:
        print('Cannot open data file - Ward Atlas Sheet 5')
    emissions_columns = ['New Code','2011.4','2011.6']
    df_emissions = pd.DataFrame(df_ward_atlas5, columns = emissions_columns)
    df_emissions.rename(columns={'New Code':'Ward_Code','2011.4':'PM10','2011.6':'NO2'},inplace=True)
    df_emissions.to_csv('files/Emissions.csv', index=False)
    return(df_emissions)

def importGreenspace():
    XLSX_WARD_ATLAS = 'https://files.datapress.com/london/dataset/ward-profiles-and-atlas/2015-09-24T14:48:35/ward-atlas-data.xls'
    try:
        df_ward_atlas5 = pd.read_excel(XLSX_WARD_ATLAS, 'iadatasheet5', skiprows=2)   
    except:
        print('Cannot open data file - Ward Atlas Sheet 5')
    greenspace_columns = ['New Code','2014.4']
    df_greenspace = pd.DataFrame(df_ward_atlas5, columns = greenspace_columns)
    df_greenspace.rename(columns={'New Code':'Ward_Code','2014.4':'Greenspace%'},inplace=True)
    df_greenspace.to_csv('files/Greenspace.csv', index=False)

def importNature():
    CSV_NATURE_ACCESS = 'https://files.datapress.com/london/dataset/access-public-open-space-and-nature-ward/public-open-space-nature-ward_access-to-nature.csv'
    try:
        df_nature_access = pd.read_csv(CSV_NATURE_ACCESS)
    except:
        print('Cannot open data file - Nature Access by Ward')
    nature_columns = ['Ward','% homes with good access to nature']
    df_nature = pd.DataFrame(df_nature_access, columns = nature_columns)
    df_nature.rename(columns={'Ward':'Ward_Code', '% homes with good access to nature':'Nature_Access'},inplace=True)
    df_nature.to_csv('files/Nature.csv', index=False)
   
def importElection():
    XLSX_ELECTION = 'https://files.datapress.com/london/dataset/london-elections-results-2016-wards-boroughs-constituency/2016-05-27T10:46:12/gla-elections-votes-all-2016.xlsx'
    try:
        df_election = pd.read_excel(XLSX_ELECTION, 'Wards & Postals', skiprows=2)   
    except:
        print('Cannot open data file - Election Turnout by Ward')
    df_turnout = df_election[(df_election['Unnamed: 2'] == 'Ward')]
    turnout_columns = ['Unnamed: 4','% Turnout']
    df_turnout = pd.DataFrame(df_turnout, columns = turnout_columns)
    df_turnout.rename(columns={'Unnamed: 4':'Ward_Code', '% Turnout':'Turnout'},inplace=True)
    df_turnout.to_csv('files/ElectionTurnout.csv', index=False)
    return(df_turnout)

def importCulturalSpace():
    df_Foursquare = pd.DataFrame()
    FrameList = []
    limit_reached = 0
    for i in range(1,50):
        for j in range(15,750,15):
            client_id = "XO1QQIQ02JE0EKQXZ4HBOYPHW2DSWJH5EIBN2AOZG2NRZVPK"
            client_secret = "GJGSIAZR4WXLYOHW4VP1JHTAMLH23EZXCWZDVPCZTYY2RQ4V"
            lat = (51.2 + (i/100.0))
            long = (-0.25 + (j/1000.0))
            category_id = '4d4b7104d754a06370d81259' # cultural space
            distance = 450
            requested_keys = ["categories","id","location","name"]
            url = "https://api.foursquare.com/v2/venues/search?ll=%s,%s&intent=browse&radius=%s&categoryId=%s&limit=49&client_id=%s&client_secret=%s&v=%s" % (lat, long, distance, category_id, client_id, client_secret, time.strftime("%Y%m%d"))
            resp = requests.get(url)
            dataResp = resp.json()
            if dataResp["response"]['venues'] != []:
                data = pd.DataFrame(dataResp["response"]['venues'])[requested_keys]
                df_FoursquareIteration = pd.DataFrame(data)
                if len(df_FoursquareIteration) == 49:
                    limit_reached = limit_reached + 1
                df_FoursquareIteration["categories"] = df_FoursquareIteration["categories"].apply(lambda x: dict(x[0])['name'])
                df_FoursquareIteration["lat"] = df_FoursquareIteration["location"].apply(lambda x: dict(x)["lat"])
                df_FoursquareIteration["long"] = df_FoursquareIteration["location"].apply(lambda x: dict(x)["lng"])
                FrameList.append(df_FoursquareIteration)
    df_Foursquare = pd.concat(FrameList)
    df_Foursquare.drop_duplicates(subset=['id'],keep=False)
    if limit_reached > 0:
        print('Limit Reached')
        print(limit_reached)
    columns = ['name','categories','lat','long']
    df_culture = pd.DataFrame(df_Foursquare, columns = columns)
    df_culture.to_csv('files/CulturalVenues.csv', index=False, encoding='utf-8')
      
def importTransportAccess():
    CSV_TRANSPORT_ACCESS = 'https://files.datapress.com/london/dataset/public-transport-accessibility-levels/2018-02-20T14:44:30.58/Ward2014%20AvPTAI2015.csv'
    try:
        df_transport_access = pd.read_csv(CSV_TRANSPORT_ACCESS)
    except:
        print('Cannot open data file - Transport Access by Ward')
    transport_access_columns = ['Ward Code','AvPTAI2015']
    df_transport_access = pd.DataFrame(df_transport_access, columns = transport_access_columns)
    df_transport_access.to_csv('files/TransportAccess.csv', index=False)
    return(df_transport_access)
    
def importJourneyTimes():
    XLS_JOURNEYTIMES = 'files/jts0501.xls'
    CSV_LSOA_MAPPING = 'https://opendata.arcgis.com/datasets/07a6d14d4a0540769f0662f4d1450bae_0.csv'
    try:
        df_journeys = pd.read_excel(XLS_JOURNEYTIMES, 'JTS0501', skiprows=6)  
    except:
        print('Cannot open data file - JourneyTimes') 
    try:
        df_lsoa_ward = pd.read_csv(CSV_LSOA_MAPPING)
    except:
        print('Cannot open data file - LSOA to Ward Mapping')
    journey_columns = ['LSOA_code','100EmpPTt','500EmpPTt','5000EmpPTt']
    df_journeys = pd.DataFrame(df_journeys, columns = journey_columns)
    df_journeys['JourneyTime'] = (df_journeys['100EmpPTt'] + df_journeys['500EmpPTt'] + df_journeys['5000EmpPTt'])/3.0
    df_journeys = pd.merge(df_journeys, df_lsoa_ward, left_on='LSOA_code', right_on='LSOA11CD')
    shapes = 'C:/Shapes/London_Ward.shp'
    wards = gpd.read_file(shapes)
    df_journeys = pd.merge(wards, df_journeys, left_on='GSS_CODE', right_on='WD15CD')
    df_journeys.to_csv('files/JourneyTimes.csv', index=False)
    
    
def importPopulationDensity():
    CSV_POP_DENSITY='https://files.datapress.com/london/dataset/land-area-and-population-density-ward-and-borough/2018-03-05T10:54:05.31/housing-density-ward.csv'
    try:
        df_pop_density = pd.read_csv(CSV_POP_DENSITY)
    except:
        print('Cannot open data file - Population Density by Ward')
    df_density = df_pop_density[(df_pop_density['Year'] == 2017)]
    density_columns = ['Code','Population_per_square_kilometre']
    df_density = pd.DataFrame(df_density, columns = density_columns)
    df_density.to_csv('files/PopDensity.csv', index=False)
    return(df_pop_density)
    
def importLifeExpectancy():
    CSV_LIFE_EX = 'https://files.datapress.com/london/dataset/life-expectancy-birth-and-age-65-ward/2016-02-09T14:25:25/life-expectancy-ward-at-Birth.csv'
    try:
        df_expectancy = pd.read_csv(CSV_LIFE_EX)
    except:
        print('Cannot open data file - Life Expectancy by Ward')
    df_life_ex = df_expectancy[(df_expectancy['Geography'] == 'Ward')]
    lifeex_columns = ['Ward','2010-14;Female;Life expectancy at birth','2010-14;Male;Life expectancy at birth']
    df_life_ex = pd.DataFrame(df_life_ex, columns = lifeex_columns)
    df_life_ex.rename(columns={'Ward':'Ward_Code', '2010-14;Female;Life expectancy at birth':'Female', '2010-14;Male;Life expectancy at birth':'Male'},inplace=True)
    df_life_ex['Life_Ex'] = (df_life_ex['Female'] + df_life_ex['Male'])/2
    del df_life_ex['Female']
    del df_life_ex['Male']
    df_life_ex.to_csv('files/LifeExpectancy.csv', index=False)
    return(df_life_ex)
    
def importChildObesity():
    XLSX_CHILD_OBESITY = 'https://files.datapress.com/london/dataset/prevalence-childhood-obesity-borough/2015-09-21T14:31:19/MSOA_Ward_LA_Obesity.xlsx'
    try:
        df_childobesity = pd.read_excel(XLSX_CHILD_OBESITY, '2011-12_2013-14', skiprows=3)     
    except:
        print('Cannot open data file - Childhood Obesity by Ward')
    df_childobesity.rename(columns={'Unnamed: 13':'Obese%'},inplace=True)
    obesity_columns = ['Geog Level','Code','LA code','Obese%']
    df_childobesity = pd.DataFrame(df_childobesity, columns = obesity_columns)
    df_obesityborough = pd.DataFrame(df_childobesity[(df_childobesity['Geog Level'] == 'LA')])
    df_obesityborough.rename(columns={'Obese%':'BoroughObese%'},inplace=True)
    borough_columns = ['Code','BoroughObese%']
    df_obesityborough = pd.DataFrame(df_obesityborough, columns = borough_columns)
    df_childobesity = pd.DataFrame(df_childobesity[(df_childobesity['Geog Level'] == 'Ward')])
    df_childobesity = pd.merge(df_childobesity, df_obesityborough, left_on='LA code', right_on='Code')
    df_childobesity['Obese%'].replace('s', df_childobesity['BoroughObese%'], inplace=True)
    ward_columns = ['Code_x', 'Obese%']
    df_childobesity = pd.DataFrame(df_childobesity, columns = ward_columns)
    df_childobesity.to_csv('files/ChildObesity.csv', index=False)
    return(df_childobesity)
    
def importIllness():
    XLSX_ILLNESS = 'https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/467775/File_8_ID_2015_Underlying_indicators.xlsx'
    CSV_LSOA_MAPPING ='https://opendata.arcgis.com/datasets/07a6d14d4a0540769f0662f4d1450bae_0.csv'
    try:
        df_illness = pd.read_excel(XLSX_ILLNESS, 'ID 2015 Health Domain')     
    except:
        print('Cannot open data file - Childhood Obesity by Ward')
    try:
        df_lsoa_ward = pd.read_csv(CSV_LSOA_MAPPING)
    except:
        print('Cannot open data file - LSOA to Ward Mapping')
    df_illness = pd.merge(df_illness, df_lsoa_ward, left_on='LSOA code (2011)', right_on='LSOA11CD')
    shapes = 'C:/Shapes/London_Ward.shp'
    wards = gpd.read_file(shapes)
    df_illness = pd.merge(wards, df_illness, left_on='GSS_CODE', right_on='WD15CD')
    df_illness.to_csv('files/Illness.csv', index=False)
    return(df_illness)
