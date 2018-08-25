"""
Created on Thu Aug 23 16:03:20 2018
@author: Ben Candy
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# define urls for csv data
XLS_HOUSE_PRICE = 'https://data.london.gov.uk/download/average-house-prices/fb8116f5-06f8-42e0-aa6c-b0b1bd69cdba/land-registry-house-prices-ward.xls'
df_HousePricesAll = pd.read_excel(XLS_HOUSE_PRICE,'Median',skiprows=[1])
df_HousePrices = pd.DataFrame(df_HousePricesAll.iloc[1:,:])
df_HousePrices['quintile'] = pd.qcut(df_HousePrices['Year ending Dec 2017'],5,labels=False)
df_HousePrices['thousands'] = df_HousePrices['Year ending Dec 2017']//1000

shapes = 'C:/Shapes/London_Ward.shp'
map_df = gpd.read_file(shapes)
merged = map_df.set_index('GSS_CODE').join(df_HousePrices.set_index('New code'),how='inner')
# set the minimum value
vmin = 0

# create the map for median values
vmax = 2000000
fig, ax = plt.subplots(1, figsize=(16, 9))
merged.plot(column='Year ending Dec 2017', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
ax.axis('off')
ax.set_title('London 2016 Median House Prices by Ward', fontdict={'fontsize': '16', 'fontweight' : '3'})
ax.annotate('Source: London Datastore, 2018',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
fig.savefig('HPMedian.png', dpi=300)

# create the map for quintiles
vmax = 4
fig, ax = plt.subplots(1, figsize=(16, 9))
merged.plot(column='quintile', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
ax.axis('off')
ax.set_title('London 2016 Median House Price Quintiles by Ward', fontdict={'fontsize': '16', 'fontweight' : '3'})
ax.annotate('Source: London Datastore, 2018',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm)
fig.savefig('HPQuintile', dpi=300)
    
