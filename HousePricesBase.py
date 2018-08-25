"""
Created on Thu Aug 23 16:03:20 2018
@author: Ben Candy
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# define urls for csv data
CSV_HOUSE_PRICE = 'https://files.datapress.com/london/dataset/average-house-prices/2018-02-19T14:39:18.66/land-registry-house-prices-ward.csv'
df_HousePricesAll = pd.read_csv(CSV_HOUSE_PRICE)
df_HousePrices = pd.DataFrame(df_HousePricesAll[(df_HousePricesAll['Measure']=='Median') & (df_HousePricesAll['Year']==2016)])
df_HousePrices['quintile'] = pd.qcut(df_HousePrices['Value'],5,labels=False)
df_HousePrices['thousands'] = df_HousePrices['Value']//1000

bplot = sns.boxplot(y=df_HousePrices['Value'])
bplot.axes.set_title('London 2016 Median House Prices by Ward',fontsize=12)
bplot.set_ylabel('Median House Price 2016',fontsize=9)
bplot.figure.savefig('HPbox.jpeg',format='jpeg',dpi=300)


def CreateChloropethMap(df,value,vmax,title,png):
    shapes = 'C:/Shapes/London_Ward.shp'
    map_df = gpd.read_file(shapes)
    merged = map_df.set_index('GSS_CODE').join(df.set_index('Code'))
    # set the minimum value
    vmin = 0
    fig, ax = plt.subplots(1, figsize=(16, 9))
    # create map
    merged.plot(column=value, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
    ax.axis('off')
    ax.set_title(title, fontdict={'fontsize': '16', 'fontweight' : '3'})
    ax.annotate('Source: London Datastore, 2016',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    fig.savefig(png, dpi=300)
    return(png)
    
CreateChloropethMap(df_HousePrices,'quintile',4,'London 2016 Median House Price Quintiles by Ward', 'HPQuintile.png')
CreateChloropethMap(df_HousePrices,'Value',2000000,'London 2016 Median House Prices by Ward', 'HPMedian.png')
