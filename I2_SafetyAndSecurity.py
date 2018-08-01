# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:10:12 2018

@author: ben.candy
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

CSV_CRIME = 'https://data.london.gov.uk/download/recorded_crime_summary/866c05de-c5cd-454b-8fe5-9e7c77ea2313/MPS%20Ward%20Level%20Crime%20%28most%20recent%2024%20months%29.csv'
XLSX_TRAFFIC_FATALITIES = 'https://files.datapress.com/london/dataset/road-casualties-severity-borough/road-casualties-severity-lsoa-msoa-ward.xls'

# import crime data
try:
    df_crime = pd.read_csv(CSV_CRIME)
except:
    print('Cannot open data file - Crime by Ward')   
    
    df_crime['2017'] = df_crime['201701']+df_crime['201702']+df_crime['201703']+df_crime['201704']+df_crime['201705']+df_crime['201706']+df_crime['201707']+df_crime['201708']+df_crime['201709']+df_crime['201710']+df_crime['201711']+df_crime['201712']
    df_crime_pivot = df_crime.pivot_table(index='WardCode', columns='Major Category', values='2017',aggfunc='sum')
    crime_columns = ['Burglary','Criminal Damage','Drugs','Robbery','Sexual Offences','Theft and Handling','Violence Against the Person']
    df_crime_2017 = pd.DataFrame(df_crime_pivot, columns = crime_columns)

    crime_scaler = StandardScaler() 
    crime_scaled_values = crime_scaler.fit_transform(df_crime_2017) 
    df_crime_2017.loc[:,:] = crime_scaled_values
    
    df_crime_cov = df_crime_2017.cov()
    df_crime_corr = df_crime_2017.corr()
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_crime_2017)
    df_crime_principal = pd.DataFrame(principalComponents, columns = ['principal component 1','principal component 2'])
        
# import election data
try:
    xlsx_traffic_fatalities = pd.ExcelFile(XLSX_TRAFFIC_FATALITIES)
    df_traffic_f = pd.read_excel(XLSX_TRAFFIC_FATALITIES, 'Ward 2014 only', skiprows=1)
except:
    print('Cannot open data file - Traffic Fatalities by Ward')
    
    traffic_columns = ['Unnamed: 0','1 Fatal.9','2 Serious.9','3 Slight.9']
    df_traffic_fatalities = pd.DataFrame(df_traffic_f, columns = traffic_columns)
    df_traffic_fatalities.rename(columns={'Unnamed: 0':'Ward_Code','1 Fatal.9':'Fatal','2 Serious.9':'Serious','3 Slight.9':'Slight'}, inplace=True)
    traffic_scaler = StandardScaler()
    traffic_scaled_values = traffic_scaler.fit_transform(df_traffic_fatalities)
    df_traffic_fatalities.loc[:,:] = traffic_scaled_values
    df_traffic_fatalities['traffic_fatality_score'] = (df_traffic_fatalities['Fatal'] + df_traffic_fatalities['Serious'] + df_traffic_fatalities['Slight'])/3
    

