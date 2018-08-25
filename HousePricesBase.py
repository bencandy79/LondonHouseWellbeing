"""
Created on Thu Aug 23 16:03:20 2018
@author: Ben Candy
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# define urls for csv data
CSV_HOUSE_PRICE = 'https://files.datapress.com/london/dataset/average-house-prices/2018-02-19T14:39:18.66/land-registry-house-prices-ward.csv'
df_HousePricesAll = pd.read_csv(CSV_HOUSE_PRICE)
df_HousePrices = df_HousePricesAll[(df_HousePricesAll['Measure']=='Median') & (df_HousePricesAll['Year']==2016)]
df_HousePrices['quintile'] = pd.qcut(df_HousePrices['Value'],5,labels=False)
df_HousePrices['thousands'] = df_HousePrices['Value']//1000

shapes = 'C:/Shapes/London_Ward.shp'
map_df = gpd.read_file(shapes)
merged = map_df.set_index('GSS_CODE').join(df_HousePrices.set_index('Code'))
variable = 'Value'
# set the range for the choropleth
vmin, vmax = 0, 2000000
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(16, 9))
# create map
merged.plot(column=variable, cmap='RdYlGn', linewidth=0.8, ax=ax, edgecolor='0.8')
# remove the axis
ax.axis('off')
# add a title
ax.set_title('London 2016 Median House Prices by Ward', fontdict={'fontsize': '18', 'fontweight' : '3'})
# create an annotation for the data source
ax.annotate('Source: London Datastore, 2016',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# empty array for the data range
sm._A = []
# add the colorbar to the figure
cbar = fig.colorbar(sm)
#fig.savefig('HPmap.png', dpi=300)
