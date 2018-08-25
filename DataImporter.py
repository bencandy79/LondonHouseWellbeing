# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:29:29 2018
@author: Ben Candy
"""
import pandas as pd
import numpy as np

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
    employment_columns = ['New Code','Economically active: % In employment']
    df_employment = pd.DataFrame(df_ward_atlas2, columns = employment_columns)
    df_employment.rename(columns={'New Code':'Ward_Code','Economically active: % In employment':'Employ_Rate'},inplace=True)
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
    CSV_SCHOOLS = 'http://ea-edubase-api-prod.azurewebsites.net/edubase/edubasealldata20180825.csv'
    try:
        iter_csv = pd.read_csv(CSV_SCHOOLS, iterator=True, chunksize=1000)
        df_schoolsall = pd.DataFrame()
        df_schoolsall = pd.concat([chunk[chunk['GOR (name)'] == 'London'] for chunk in iter_csv])
    except:
        print('Cannot open data file - SchoolsData')
    df_schoolslondonopen = df_schoolsall[(df_schoolsall['SchoolCapacity'] > 0) & (df_schoolsall['EstablishmentStatus (code)'] == 1)]
    df_schoolslondonopen['OfstedRating (name)'].replace('', np.nan, inplace=True)
    df_schoolslondon = pd.DataFrame(df_schoolslondonopen.dropna(subset=['OfstedRating (name)'], inplace=True))
    df_schoolslondon['Ofsted_Rating'] = df_schoolslondon['OfstedRating (name)']
    ofsted_ratings = {'Ofsted_Rating':{'Outstanding':2,'Good':1,'Requires improvement':-1,'Serious Weaknesses':-2,'Special Measures':-2}}
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
        print('Cannot open data file - Ward Atlas Sheet 2')
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
        print('Cannot open data file - Ward Atlas Sheet 2')
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
