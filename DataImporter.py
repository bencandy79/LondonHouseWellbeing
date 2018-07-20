"""
IMPORT OPEN SOURCE CSV DATA

@author: Ben Candy
"""
# This approach minimises storage requirements and means that application can work from any internet-enabled device

import pandas as pd

# define urls for csv data
CSV_HOUSE_PRICE = 'https://files.datapress.com/london/dataset/average-house-prices/2018-02-19T14:39:18.66/land-registry-house-prices-ward.csv'
CSV_CRIME = 'https://data.london.gov.uk/download/recorded_crime_summary/866c05de-c5cd-454b-8fe5-9e7c77ea2313/MPS%20Ward%20Level%20Crime%20%28most%20recent%2024%20months%29.csv'
CSV_TRANSPORT_ACCESS = 'https://files.datapress.com/london/dataset/public-transport-accessibility-levels/2018-02-20T14:44:30.58/Ward2014%20AvPTAI2015.csv'
CSV_OPENSPACE_ACCESS = 'https://files.datapress.com/london/dataset/access-public-open-space-and-nature-ward/public-open-space-nature-ward_access-to-nature.csv'
CSV_POP_DENSITY = 'https://files.datapress.com/london/dataset/land-area-and-population-density-ward-and-borough/2018-03-05T10:54:05.31/housing-density-ward.csv'
CSV_HOUSEHOLD_INCOME = 'https://files.datapress.com/london/dataset/household-income-estimates-small-areas/modelled-household-income-estimates-wards.csv'
CSV_LIFE_EXPECTANCY = 'https://files.datapress.com/london/dataset/life-expectancy-birth-and-age-65-ward/2016-02-09T14:25:25/life-expectancy-ward-at-Birth.csv'
XLSX_2016_ELECTION = 'https://files.datapress.com/london/dataset/london-elections-results-2016-wards-boroughs-constituency/2016-05-27T10:46:12/gla-elections-votes-all-2016.xlsx'

# import house prices
try:
    df_house_price = pd.read_csv(CSV_HOUSE_PRICE)
except:
    print('Cannot open data file - Prices by Ward')
    print(CSV_HOUSE_PRICE)
  
# import crime data
try:
    df_crime = pd.read_csv(CSV_CRIME)
except:
    print('Cannot open data file - Crime by Ward')
    print(CSV_CRIME)

# import transport access data
try:
    df_transport_access = pd.read_csv(CSV_TRANSPORT_ACCESS)
except:
    print('Cannot open data file - Transport Access by Ward')
    print(CSV_TRANSPORT_ACCESS)

# import open space and nature access data
try:
    df_openspace_access = pd.read_csv(CSV_OPENSPACE_ACCESS)
except:
    print('Cannot open data file - Open Space and Nature Access by Ward')
    print(CSV_OPENSPACE_ACCESS)

# import population density data
try:
    df_pop_density = pd.read_csv(CSV_POP_DENSITY)
except:
    print('Cannot open data file - Population Density by Ward')
    print(CSV_POP_DENSITY)

# import household income data
try:
    df_household_income = pd.read_csv(CSV_HOUSEHOLD_INCOME)
except:
    print('Cannot open data file - Household Income by Ward')
    print(CSV_HOUSEHOLD_INCOME)
    
# import life expectancy data
try:
    df_life_expectancy = pd.read_csv(CSV_LIFE_EXPECTANCY)
except:
    print('Cannot open data file - Life Expectancy by Ward')
    print(CSV_LIFE_EXPECTANCY)
    
# import election data
try:
    df_election = pd.read_excel(XLSX_2016_ELECTION, sheet_name=1, skiprows=2)
except:
    print('Cannot open data file - Election 2016 by Ward')
    print(XLSX_2016_ELECTION)






