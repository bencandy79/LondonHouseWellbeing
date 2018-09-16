# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:39:02 2018
@author: Ben Candy
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
import shapely

def standardise(df):
    # standardises values in a dataframe and returns new values to the dataframe
    STscaler = StandardScaler()
    scaled_values = STscaler.fit_transform(df)
    df.loc[:,:] = scaled_values
    return(df)
    
def tablestyle(df):
    th = [('font-size','11px'),('text-align','centre'),('font-weight','bold')
    ,('color','#6d6d6d'),('background-color','#f7f7f9')] 
    td = [('font-size','11px')]
    styles = [dict(selector='th',props=th),dict(selector='td',props=td)]
    df.style.set_table_styles(styles)
    return(df)

# calculates domain score (out of 100) using Principal Component Analysis
# used where there is significant covariance for the three indicators    
def domain_indicator_PCA(df, indicator_name):
    # function runs PCA on 3 indicators in a domain and returns a score for domain out of 100
    NMscaler = MinMaxScaler()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    df['Indicator'] = pd.Series(principalComponents[:,0],index=df.index)
    normComponents = NMscaler.fit_transform(df)
    df_norm = df
    df_norm.loc[:,:] = normComponents
    indicator = df_norm.loc[:,'Indicator']*100
    indicator.rename(columns={'Indicator':indicator_name},inplace=True)
    return(indicator)

# calculates domain score (out of 100) using arithmetic mean
# used where the three indicators have low covariance    
def domain_indicator_Mean(df, indicator_name):
    df['Indicator'] = (df.iloc[:,0] + df.iloc[:,1] + df.iloc[:,2])/3
    NMscaler = MinMaxScaler()
    normComponents = NMscaler.fit_transform(df)
    df_norm = df
    df_norm.loc[:,:] = normComponents
    indicator = df_norm.loc[:,'Indicator']*100
    indicator.rename(columns={'Indicator':indicator_name},inplace=True)
    return(indicator)

# maps points to London ward polygons - projects lat long point co-ords to OS map grid co-ords   
def mapLatLongToWard(df):
    # takes latitude and longditude of a place
    # converts co-ordinate reference system to match shapefile
    # maps point to ward polygon
    inProj = Proj(init='epsg:4326') # lat long
    outProj = Proj(init='epsg:27700') # OS National Grid
    df['points'] = df.apply(lambda row: Point(transform(inProj, outProj, row['long'],row['lat'])), axis=1)
    del (df['lat'], df['long'])
    df = gpd.GeoDataFrame(df, geometry='points')
    shapes = 'datastore/Shapes/London_Ward.shp'
    wards = gpd.GeoDataFrame.from_file(shapes)
    wards = wards[['GSS_CODE','geometry']]
    df = gpd.tools.sjoin(df, wards, how='left')
    df.dropna(subset=['index_right'], inplace=True)
    return(df)

# maps pre-2014 wards to best-fit post-2014 wards
# affects pre-2014 wards in Hackney, Kensington & Chelsea and Tower Hamlets
def old_to_new_wards(df,column_name):
    new_wards = ['E05009367','E05009368','E05009369','E05009370','E05009371','E05009372','E05009373',
                 'E05009374','E05009375','E05009376','E05009377','E05009378','E05009379','E05009380',
                 'E05009381','E05009382','E05009383','E05009384','E05009385','E05009386','E05009387',
                 'E05009388','E05009389','E05009390','E05009391','E05009392','E05009393','E05009394',
                 'E05009395','E05009396','E05009397','E05009398','E05009399','E05009400','E05009401',
                 'E05009402','E05009403','E05009404','E05009405','E05009317','E05009318','E05009319',
                 'E05009320','E05009321','E05009322','E05009323','E05009324','E05009325','E05009326',
                 'E05009327','E05009328','E05009332','E05009333','E05009329','E05009330','E05009331',
                 'E05009334','E05009335','E05009336']
    old_wards = ['E05000231','E05000232','E05000234','E05000235','E05000236','E05000237','E05000238',
                 'E05000249','E05000239','E05000233','E05000239','E05000236','E05000241','E05000242',
                 'E05000245','E05000235','E05000246','E05000243','E05000247','E05000248','E05000244',
                 'E05000382','E05000383','E05000384','E05000387','E05000385','E05000386','E05000389',
                 'E05000388','E05000389','E05000391','E05000392','E05000393','E05000394','E05000395',
                 'E05000396','E05000397','E05000398','E05000399','E05000581','E05000575','E05000576',
                 'E05000577','E05000578','E05000578','E05000583','E05000583','E05000579','E05000580',
                 'E05000582','E05000575','E05000586','E05000587','E05000584','E05000585','E05000573',
                 'E05000584','E05000588','E05000589']
    frames = []
    df_newvalues = pd.DataFrame()
    for i in range(0,len(new_wards)):
        new_frame = pd.DataFrame(df.loc[old_wards[i],:])
        new_frame.rename(columns={old_wards[i]:column_name},inplace=True)
        new_frame['ward'] = new_wards[i]
        new_frame.set_index('ward', inplace=True)
        frames.append(new_frame)
    df_newvalues = pd.concat(frames, sort=False)
    final_frames = [df, df_newvalues]
    df_newwards = pd.concat(final_frames, sort=False)
    return(df_newwards)

# function to transform ordnance survey polygon geometry to lat long co-ords
# required when passing to javaScript visualisation    
def geometricTransform(gdf):
    point_list = []
    poly_list = []
    for index, row in gdf.iterrows():
        for pt in list(row['geometry'].exterior.coords):
              inProj = Proj(init='epsg:27700') # lat long
              outProj = Proj(init='epsg:4326') # OS National Grid
              new_Point = transform(inProj, outProj, pt[0], pt[1])
              point_list.append(new_Point)
        poly = shapely.geometry.Polygon(point_list)
        point_list = []
        poly_list.append(poly)
    gdf['geometry'] = gpd.GeoSeries(poly_list)
    return(gdf)

# output geoJSON to javascript file defining a js variable for interactive map
def toGeoJSON(gdf):
    json_file = gdf.to_json()
    with open('dataset.js', 'w') as openfile:
        openfile.write('var dataset = ' + json_file + ';')

# house prices
df_HousePrices = pd.read_csv('datastore/HousePrices.csv')
df_HousePrices['quintile'] = pd.qcut(df_HousePrices['Year ending Dec 2017'],5,labels=False)
df_HousePrices['thousands'] = df_HousePrices['Year ending Dec 2017']//1000
house_price_columns = ['New code', 'Ward name', 'Borough name', 'Year ending Dec 2017', 'quintile','thousands']
df_HousePrices = pd.DataFrame(df_HousePrices, columns = house_price_columns)
    
# read shape file for ward polygons
shapes = 'datastore/Shapes/London_Ward.shp'
map_df = gpd.read_file(shapes)
    
# Domain One - Education and Employment
# employment  
df_employment = pd.read_csv('datastore/Employment.csv')
df_employment.set_index('Ward_Code',inplace=True)
df_employment = df_employment.drop(['K04000001'])
df_employment = df_employment.drop(['E92000001'])
df_employment = df_employment.drop(['E12000007'])
df_employment = standardise(df_employment)
df_employment = old_to_new_wards(df_employment,'Employ_Rate')
# average GCSE scores
df_GCSE = pd.read_csv('datastore/GCSE.csv')
df_GCSE.set_index('Ward_Code',inplace=True)
df_GCSE = df_GCSE.drop(['K04000001'])
df_GCSE = df_GCSE.drop(['E92000001'])
df_GCSE = df_GCSE.drop(['E12000007'])
df_GCSE = standardise(df_GCSE)
df_GCSE = old_to_new_wards(df_GCSE,'Av_GCSE_Points')
# average Ofsted rating for schools
df_schoolslondonopen = pd.read_csv('datastore/Schools.csv')
df_schoolsborough = df_schoolslondonopen.pivot_table(values='Ofsted_Rating', index='DistrictAdministrative (code)', fill_value=0)
df_schools = df_schoolslondonopen.pivot_table(values='Ofsted_Rating', index='AdministrativeWard (code)', fill_value=0)
df_schoolward = pd.merge(map_df, df_schools, left_on='GSS_CODE', right_index=True, how='outer')
df_schools = pd.merge(df_schoolward,df_schoolsborough, left_on='LB_GSS_CD', right_index=True, how='outer')
df_schools['Ofsted_Rating_x'].fillna(df_schools['Ofsted_Rating_y'], inplace=True)
final_columns = ['GSS_CODE','Ofsted_Rating_x']
df_schools = pd.DataFrame(df_schools, columns = final_columns)
df_schools.set_index('GSS_CODE',inplace=True)
df_schools = standardise(df_schools)
# merge the 3 indicators to create the domain
df_edu = pd.merge(df_GCSE,df_schools, left_index=True, right_index=True)
df_eduemploy = pd.merge(df_employment,df_edu, left_index=True, right_index=True)
# check covariance and correlation matrices
df_eduemploy_cov = df_eduemploy.cov()
df_eduemploy_corr = df_eduemploy.corr()
df_eduemploy_corr.to_csv('datastore/eduemploycorr.csv')
## run principal component analysis to create domain score
Education_Employment = domain_indicator_Mean(df_eduemploy,'Education_Employment')

# Domain Two - Safety and Security
# crime (against the person and locality crime)
df_crime = pd.read_csv('datastore/Crime.csv')
df_crime_pivot = df_crime.pivot_table(index='WardCode', columns='Major Category', values='2017',aggfunc='sum')
person_columns = ['Robbery','Sexual Offences','Violence Against the Person']
area_columns = ['Burglary','Criminal Damage','Drugs','Theft and Handling']
df_crimeperson = pd.DataFrame(df_crime_pivot, columns = person_columns)
df_crimeperson = standardise(df_crimeperson)
df_crimearea = pd.DataFrame(df_crime_pivot, columns = area_columns)
df_crimearea = standardise(df_crimearea)
# traffic danger
df_traffic = pd.read_csv('datastore/Traffic.csv')
df_traffic.set_index('Ward_Code',inplace=True)
df_traffic = standardise(df_traffic)
df_traffic['traffic_fatality_score'] = (df_traffic['Fatal'] + df_traffic['Serious'] + df_traffic['Slight'])/3
del df_traffic['Fatal']
del df_traffic['Serious']
del df_traffic['Slight']
# merge the 3 indicators to create the domain
df_crime = pd.merge(df_crimeperson, df_crimearea, left_index=True, right_index=True)
df_safetysecurity = pd.merge(df_crime,df_traffic, left_index=True, right_index=True)
# check covariance and correlation matrices
df_safetysecurity_cov = df_safetysecurity.cov()
df_safetysecurity_corr = df_safetysecurity.corr()
df_safetysecurity_corr.to_csv('datastore/safetysecuritycorr.csv')
# run principal component analysis to create domain score
Safety_Security = 100-domain_indicator_PCA(df_safetysecurity,'Safety_Security')

# Domain Three - Environment
# emissions
df_emissions = pd.read_csv('datastore/Emissions.csv')
df_emissions.set_index('Ward_Code',inplace=True)
df_emissions = df_emissions.drop(['K04000001'])
df_emissions = df_emissions.drop(['E92000001'])
df_emissions = df_emissions.drop(['E12000007'])
df_emissions = standardise(df_emissions)
df_emissions['emission'] = (df_emissions['PM10'] + df_emissions['NO2'])/2
df_emissions = pd.DataFrame(df_emissions['emission']*-1) # emissions negative so *-1
df_emissions = old_to_new_wards(df_emissions,'emission')
# greenspace
df_greenspace = pd.read_csv('datastore/Greenspace.csv')
df_greenspace.set_index('Ward_Code',inplace=True)
df_greenspace = df_greenspace.drop(['K04000001'])
df_greenspace = df_greenspace.drop(['E92000001'])
df_greenspace = df_greenspace.drop(['E12000007'])
df_greenspace = standardise(df_greenspace)
df_greenspace = old_to_new_wards(df_greenspace,'Greenspace%')
# nature
df_nature = pd.read_csv('datastore/Nature.csv')
df_nature.set_index('Ward_Code',inplace=True)
df_nature = standardise(df_nature)
df_nature = old_to_new_wards(df_nature,'Nature_Access')
# merge the 3 indicators to create the domain
df_green = pd.merge(df_nature,df_greenspace, left_index=True, right_index=True)
df_environment = pd.merge(df_green,df_emissions, left_index=True, right_index=True)
# check covariance and correlation matrices
df_environment_cov = df_environment.cov()
df_environment_corr = df_environment.corr()
df_environment_corr.to_csv('datastore/environmentcorr.csv')
# create domain indicator using arithmetic mean
Environment = domain_indicator_Mean(df_environment,'Environment')

# Domain Four - Community Vitality and Participation
# election turnout
df_turnout = pd.read_csv('datastore/ElectionTurnout.csv')
df_turnout.set_index('Ward_Code', inplace=True)
df_turnout = standardise(df_turnout)
# access to cultural spaces
cultural_venues = pd.read_csv('datastore/CulturalVenues.csv')
cultural_locations = mapLatLongToWard(cultural_venues)
df_culturepivot = cultural_locations.pivot_table(index='GSS_CODE', values='name',aggfunc='count')
df_cultureaccess = standardise(df_culturepivot)
# access to food and drink establishment
bars = pd.read_csv('datastore/Bars.csv')
restaurants = pd.read_csv('datastore/Restaurants.csv')
food_drink_list = [bars, restaurants]
bars_restaurants = pd.concat(food_drink_list)
# specify co-ordinate reference systems
bar_restaurant_locations = mapLatLongToWard(bars_restaurants)
df_barrestaurantpivot = bar_restaurant_locations.pivot_table(index='GSS_CODE', values='name',aggfunc='count')
df_barrestaurant = standardise(df_barrestaurantpivot)
df_barrestaurant.rename(columns={'name':'name_y'},inplace=True)
# merge the 3 indicators to create the domain
df_participation = df_turnout.join(df_cultureaccess, how='outer')
df_vitalityparticipation = df_participation.join(df_barrestaurant, how='outer')
df_vitalityparticipation['name'].fillna(0.0, inplace=True)
df_vitalityparticipation['name_y'].fillna(0.0, inplace=True)
df_vitalityparticipation['Turnout'].fillna(0.0, inplace=True)
# check covariance and correlation matrices
df_vitalityparticipation_cov = df_vitalityparticipation.cov()
df_vitalityparticipation_corr = df_vitalityparticipation.corr()
df_vitalityparticipation_corr.to_csv('datastore/vitalityparticipationcorr.csv')
# create domain indicator using arithmetic mean
Vitality_Participation = domain_indicator_Mean(df_vitalityparticipation,'Vitality_Participation')

# Domain Five - Infrastructure
# access to public transportation
df_pta = pd.read_csv('datastore/TransportAccess.csv')
df_pta.set_index('Ward Code',inplace=True)
df_pta = standardise(df_pta)
# average journey times
df_journeys = pd.read_csv('datastore/JourneyTimes.csv')
df_journeypivot = df_journeys.pivot_table(index='GSS_CODE', values='JourneyTime')
df_journeys = standardise(df_journeypivot)*-1
# population density
df_density = pd.read_csv('datastore/PopDensity.csv')
df_density.set_index('Code',inplace=True)
df_density = standardise(df_density)*-1
df_density = old_to_new_wards(df_density,'Population_per_square_kilometre')
# merge the 3 indicators to create the domain
df_transportation = pd.merge(df_pta,df_journeys, left_index=True, right_index=True)
df_infrastructure = pd.merge(df_transportation,df_density, left_index=True, right_index=True)
# check covariance and correlation matrices
df_infrastructure_cov = df_infrastructure.cov()
df_infrastructure_corr = df_infrastructure.corr()
df_infrastructure_corr.to_csv('datastore/infrastructurecorr.csv')
# create domain indicator using PCA
Infrastructure = domain_indicator_PCA(df_infrastructure,'Infrastructure')

# Domain Six - Health
# life expectancy
df_expectancy = pd.read_csv('datastore/LifeExpectancy.csv')
df_expectancy.set_index('Ward_Code', inplace=True)
df_expectancy = standardise(df_expectancy)
df_expectancy = old_to_new_wards(df_expectancy,'Life_Ex')
# childhood obesity
df_obesity = pd.read_csv('datastore/ChildObesity.csv')
df_obesity.set_index('Code_x', inplace=True)
df_obesity = standardise(df_obesity)*-1
# illness affecting work
df_illness = pd.read_csv('datastore/Illness.csv')
df_illnesspivot = df_illness.pivot_table(index='GSS_CODE',values='Comparative illness and disability ratio indicator')
df_illness = standardise(df_illnesspivot)*-1
# merge the 3 indicators to create the domain
df_prehealth = pd.merge(df_illness,df_obesity, left_index=True, right_index=True)
df_health = pd.merge(df_prehealth,df_expectancy, left_index=True, right_index=True)
# check covariance and correlation matrices
df_health_cov = df_health.cov()
df_health_corr = df_health.corr()
df_health_corr.to_csv('datastore/healthcorr.csv')
# build domain indicator using PCA
Health = domain_indicator_PCA(df_health,'Health')

# build data set for modelling
DataSet = pd.merge(df_HousePrices, pd.DataFrame(Education_Employment), left_on = 'New code', right_index = True)
DataSet.rename(columns={0:'Education_Employment'},inplace=True)
DataSet = pd.merge(DataSet, pd.DataFrame(Safety_Security), left_on = 'New code', right_index = True)
DataSet.rename(columns={0:'Safety_Security'},inplace=True)
DataSet = pd.merge(DataSet, pd.DataFrame(Environment), left_on = 'New code', right_index = True)
DataSet.rename(columns={0:'Environment'},inplace=True)
DataSet = pd.merge(DataSet, pd.DataFrame(Vitality_Participation), left_on = 'New code', right_index = True)
DataSet.rename(columns={0:'Vitality_Participation'},inplace=True)
DataSet = pd.merge(DataSet, pd.DataFrame(Infrastructure), left_on = 'New code', right_index = True)
DataSet.rename(columns={0:'Infrastructure'},inplace=True)
DataSet = pd.merge(DataSet, pd.DataFrame(Health), left_on = 'New code', right_index = True)
DataSet.rename(columns={0:'Health','Year ending Dec 2017':'Median_Price'},inplace=True)
DataSet.to_csv('datastore/DataSet.csv', index=False)

df_predictions = pd.read_csv('datastore/predictions.csv')

df_datasetmapping = gpd.GeoDataFrame(pd.merge(map_df, df_predictions, left_on='GSS_CODE', right_on='New code'))

df_datasetmapping.to_csv('datastore/datasetmapping.csv')

df_js = geometricTransform(df_datasetmapping)

toGeoJSON(df_js)

 
