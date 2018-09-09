
import shapely
import geopandas as gpd
from pyproj import Proj, transform


shapes = 'C:/Shapes/London_Ward.shp'
gdf = gpd.GeoDataFrame(gpd.read_file(shapes))

def geometricTransform():
    point_list = []
    poly_list = []
    for index, row in gdf.iterrows():
        for pt in list(row['geometry'].exterior.coords):
              inProj = Proj(init='epsg:27700') # lat long
              outProj = Proj(init='epsg:4326') # OS National Grid
              new_Point = transform(inProj, outProj, pt[0], pt[1])
              #new_Point_List.append(new_Point[0])
              #new_Point_List.append(new_Point[1])
              point_list.append(new_Point)
        poly = shapely.geometry.Polygon(point_list)
        point_list = []
        poly_list.append(poly)
    gdf['geometry'] = gpd.GeoSeries(poly_list)
    return(gdf)

for_json = geometricTransform(gdf)
#    
json_file = gdf.to_json()
##
with open('dataset.js', 'w') as openfile:
    openfile.write('var dataset = ' + json_file + ';')
          
          
     

    
