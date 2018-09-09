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
    
def domain_indicator(df, indicator_name):
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
    
def mapLatLongToWard(df):
    inProj = Proj(init='epsg:4326') # lat long
    outProj = Proj(init='epsg:27700') # OS National Grid
    df['points'] = df.apply(lambda row: Point(transform(inProj, outProj, row['long'],row['lat'])), axis=1)
    del (df['lat'], df['long'])
    df = gpd.GeoDataFrame(df, geometry='points')
    shapes = 'C:/Shapes/London_Ward.shp'
    wards = gpd.GeoDataFrame.from_file(shapes)
    wards = wards[['GSS_CODE','geometry']]
    df = gpd.tools.sjoin(df, wards, how='left')
    df.dropna(subset=['index_right'], inplace=True)
    return(df)

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
    
def df_to_geojson(df, properties):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in df.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry': shapely.geometry.mapping(df['geometry'])}
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson

# house prices
df_HousePrices = pd.read_csv('files/HousePrices.csv')
df_HousePrices['quintile'] = pd.qcut(df_HousePrices['Year ending Dec 2017'],5,labels=False)
df_HousePrices['thousands'] = df_HousePrices['Year ending Dec 2017']//1000
house_price_columns = ['New code', 'Ward name', 'Borough name', 'Year ending Dec 2017', 'quintile','thousands']
df_HousePrices = pd.DataFrame(df_HousePrices, columns = house_price_columns)
    
# read shape file for wards
shapes = 'C:/Shapes/London_Ward.shp'
map_df = gpd.read_file(shapes)
    
# Domain One - Education and Employment
    
df_employment = pd.read_csv('files/Employment.csv')
df_employment.set_index('Ward_Code',inplace=True)
df_employment = df_employment.drop(['K04000001'])
df_employment = df_employment.drop(['E92000001'])
df_employment = df_employment.drop(['E12000007'])
df_employment = standardise(df_employment)
df_employment = old_to_new_wards(df_employment,'Employ_Rate')

df_GCSE = pd.read_csv('files/GCSE.csv')
df_GCSE.set_index('Ward_Code',inplace=True)
df_GCSE = df_GCSE.drop(['K04000001'])
df_GCSE = df_GCSE.drop(['E92000001'])
df_GCSE = df_GCSE.drop(['E12000007'])
df_GCSE = standardise(df_GCSE)
df_GCSE = old_to_new_wards(df_GCSE,'Av_GCSE_Points')

df_schoolslondonopen = pd.read_csv('files/Schools.csv')
df_schoolsborough = df_schoolslondonopen.pivot_table(values='Ofsted_Rating', index='DistrictAdministrative (code)', fill_value=0)
df_schools = df_schoolslondonopen.pivot_table(values='Ofsted_Rating', index='AdministrativeWard (code)', fill_value=0)
df_schoolward = pd.merge(map_df, df_schools, left_on='GSS_CODE', right_index=True, how='outer')
df_schools = pd.merge(df_schoolward,df_schoolsborough, left_on='LB_GSS_CD', right_index=True, how='outer')
df_schools['Ofsted_Rating_x'].fillna(df_schools['Ofsted_Rating_y'], inplace=True)
final_columns = ['GSS_CODE','Ofsted_Rating_x']
df_schools = pd.DataFrame(df_schools, columns = final_columns)
df_schools.set_index('GSS_CODE',inplace=True)
df_schools = standardise(df_schools)
#
## merge the 3 indicators to create the domain
df_edu = pd.merge(df_GCSE,df_schools, left_index=True, right_index=True)
df_eduemploy = pd.merge(df_employment,df_edu, left_index=True, right_index=True)
#
## check covriance and correlation matrices
df_eduemploy_cov = df_eduemploy.cov()
df_eduemploy_corr = df_eduemploy.corr()
#
## run principal component analysis to create domain score
Education_Employment = domain_indicator(df_eduemploy,'Education_Employment')


# Domain Two - Safety and Security

df_crime = pd.read_csv('files/Crime.csv')
df_crime_pivot = df_crime.pivot_table(index='WardCode', columns='Major Category', values='2017',aggfunc='sum')
person_columns = ['Robbery','Sexual Offences','Violence Against the Person']
area_columns = ['Burglary','Criminal Damage','Drugs','Theft and Handling']
df_crimeperson = pd.DataFrame(df_crime_pivot, columns = person_columns)
df_crimeperson = standardise(df_crimeperson)
df_crimearea = pd.DataFrame(df_crime_pivot, columns = area_columns)
df_crimearea = standardise(df_crimearea)

df_traffic = pd.read_csv('files/Traffic.csv')
df_traffic.set_index('Ward_Code',inplace=True)
df_traffic = standardise(df_traffic)
df_traffic['traffic_fatality_score'] = (df_traffic['Fatal'] + df_traffic['Serious'] + df_traffic['Slight'])/3
del df_traffic['Fatal']
del df_traffic['Serious']
del df_traffic['Slight']

# merge the 3 indicators to create the domain
df_crime = pd.merge(df_crimeperson, df_crimearea, left_index=True, right_index=True)
df_safetysecurity = pd.merge(df_crime,df_traffic, left_index=True, right_index=True)

# check covriance and correlation matrices
df_safetysecurity_cov = df_safetysecurity.cov()
df_safetysecurity_corr = df_safetysecurity.corr()

# run principal component analysis to create domain score
Safety_Security = 100-domain_indicator(df_safetysecurity,'Safety_Security')


# Domain Three - Environment

df_emissions = pd.read_csv('files/Emissions.csv')
df_emissions.set_index('Ward_Code',inplace=True)
df_emissions = df_emissions.drop(['K04000001'])
df_emissions = df_emissions.drop(['E92000001'])
df_emissions = df_emissions.drop(['E12000007'])
df_emissions = standardise(df_emissions)
df_emissions['emission'] = (df_emissions['PM10'] + df_emissions['NO2'])/2
df_emissions = pd.DataFrame(df_emissions['emission']*-1) # emissions negative so *-1
df_emissions = old_to_new_wards(df_emissions,'emission')

df_greenspace = pd.read_csv('files/Greenspace.csv')
df_greenspace.set_index('Ward_Code',inplace=True)
df_greenspace = df_greenspace.drop(['K04000001'])
df_greenspace = df_greenspace.drop(['E92000001'])
df_greenspace = df_greenspace.drop(['E12000007'])
df_greenspace = standardise(df_greenspace)
df_greenspace = old_to_new_wards(df_greenspace,'Greenspace%')

df_nature = pd.read_csv('files/Nature.csv')
df_nature.set_index('Ward_Code',inplace=True)
df_nature = standardise(df_nature)
df_nature = old_to_new_wards(df_nature,'Nature_Access')

df_green = pd.merge(df_nature,df_greenspace, left_index=True, right_index=True)
df_environment = pd.merge(df_green,df_emissions, left_index=True, right_index=True)

df_environment_cov = df_environment.cov()
df_environment_corr = df_environment.corr()

Environment = domain_indicator(df_environment,'Environment')


# Domain Four - Community Vitality and Participation

df_turnout = pd.read_csv('files/ElectionTurnout.csv')
df_turnout.set_index('Ward_Code', inplace=True)
df_turnout = standardise(df_turnout)

positions = pd.read_csv('files/CulturalVenues.csv')
# specify co-ordinate reference systems
locations = mapLatLongToWard(positions)
df_culturepivot = locations.pivot_table(index='GSS_CODE', values='name',aggfunc='count')
df_cultureaccess = standardise(df_culturepivot)

df_vitalityparticipation = df_turnout.join(df_cultureaccess,how='outer')
df_vitalityparticipation['name'].fillna(0.0, inplace=True)
df_vitalityparticipation['Turnout'].fillna(0.0, inplace=True)

df_vitalityparticipation_cov = df_vitalityparticipation.cov()
df_vitalityparticipation_corr = df_vitalityparticipation.corr()

Vitality_Participation = domain_indicator(df_vitalityparticipation,'Vitality_Participation')

# Domain Five - Infrastructure

df_pta = pd.read_csv('files/TransportAccess.csv')
df_pta.set_index('Ward Code',inplace=True)
df_pta = standardise(df_pta)

df_journeys = pd.read_csv('files/JourneyTimes.csv')
df_journeypivot = df_journeys.pivot_table(index='GSS_CODE', values='JourneyTime')
df_journeys = standardise(df_journeypivot)*-1

df_density = pd.read_csv('files/PopDensity.csv')
df_density.set_index('Code',inplace=True)
df_density = standardise(df_density)*-1
df_density = old_to_new_wards(df_density,'Population_per_square_kilometre')

df_transportation = pd.merge(df_pta,df_journeys, left_index=True, right_index=True)
df_infrastructure = pd.merge(df_transportation,df_density, left_index=True, right_index=True)

df_infrastructure_cov = df_infrastructure.cov()
df_infrastructure_corr = df_infrastructure.corr()

Infrastructure = domain_indicator(df_infrastructure,'Infrastructure')


# Domain Six - Health

df_expectancy = pd.read_csv('files/LifeExpectancy.csv')
df_expectancy.set_index('Ward_Code', inplace=True)
df_expectancy = standardise(df_expectancy)
df_expectancy = old_to_new_wards(df_expectancy,'Life_Ex')

df_obesity = pd.read_csv('files/ChildObesity.csv')
df_obesity.set_index('Code_x', inplace=True)
df_obesity = standardise(df_obesity)*-1

df_illness = pd.read_csv('files/Illness.csv')
df_illnesspivot = df_illness.pivot_table(index='GSS_CODE',values='Comparative illness and disability ratio indicator')
df_illness = standardise(df_illnesspivot)*-1

df_prehealth = pd.merge(df_illness,df_obesity, left_index=True, right_index=True)
df_health = pd.merge(df_prehealth,df_expectancy, left_index=True, right_index=True)

df_health_cov = df_health.cov()
df_health_corr = df_health.corr()

Health = domain_indicator(df_health,'Health')


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


DataSet.to_csv('files/DataSet.csv', index=False)

map_reproj = map_df.copy()
map_reproj['geometry'] = map_reproj['geometry'].to_crs(epsg=4326)
bigmap = df_to_geojson(map_reproj, ['NAME','HECTARES'])


    
#1dcb42e6633b78bf825f8a79fd05560deeed49c7  
