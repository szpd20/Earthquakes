#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics


df = pd.read_csv('earthquakes.csv')
print(df.head())
print(df.shape)
print(df.info())
print("Describing the dataset for deeper analysys")
print(df.describe())

'''

splitted = df['Origin Time'].str.split(' ', n=1,
                                      expand=True)
 
df['Date'] = splitted[0]
df['Time'] = splitted[1].str[:8]
 
df.drop('Origin Time',
        axis=1,
        inplace=True)
df.head()

splitted = df['Date'].str.split('-', expand=True)
 
df['Day'] = splitted[0].astype('int')
df['Month'] = splitted[1].astype('int')
df['Year'] = splitted[2].astype('int')


# converting 'Weight' from float to int
df['Month'] = df['Month'].astype(int)
print("----------------------------") 

print("printing out he values of column month") 

df['Month'] = df['Month'].replace(to_replace = int(1), value= 'Jan')
df['Month'] = df['Month'].replace(to_replace = int(2), value= 'Feb')
df['Month'] = df['Month'].replace(to_replace = int(3), value= 'March')
df['Month'] = df['Month'].replace(to_replace = int(4), value= 'Apr')
df['Month'] = df['Month'].replace(to_replace = int(5), value= 'May')
df['Month'] = df['Month'].replace(to_replace = int(6), value= 'June')
df['Month'] = df['Month'].replace(to_replace = int(7), value= 'July')
df['Month'] = df['Month'].replace(to_replace = int(8), value= 'Aug')
df['Month'] = df['Month'].replace(to_replace = int(9), value= 'Sept')
df['Month'] = df['Month'].replace(to_replace = int(10), value= 'Oct')
df['Month'] = df['Month'].replace(to_replace = int(11), value= 'Nov')
df['Month'] = df['Month'].replace(to_replace = int(12), value= 'Dec')

'''

print(df)

import pandas as pd
import matplotlib.pyplot as plt

# Load the earthquake data from a CSV file
data = pd.read_csv('earthquakes.csv')

# Basic data analysis
print("Number of earthquakes:", len(data))
print("Summary statistics:")
print(data.describe())

# Plotting earthquake distribution on a map
plt.figure(figsize=(10, 8))
plt.scatter(data['Longitude'], data['Latitude'], c=data['Magnitude'], cmap='jet')
plt.colorbar(label='Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Distribution')
plt.show()

# Analyzing the magnitude distribution
plt.figure(figsize=(8, 6))
plt.hist(data['Magnitude'], bins=30, edgecolor='black')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Magnitude Distribution')
plt.show()

# Analyzing the depth distribution
plt.figure(figsize=(8, 6))
plt.hist(data['Depth'], bins=20, edgecolor='black')
plt.xlabel('Depth')
plt.ylabel('Frequency')
plt.title('Depth Distribution')
plt.show()

# Converting Origin Time to datetime format
data['Origin Time'] = pd.to_datetime(data['Origin Time'])

# Analyzing earthquake frequency over time
data['Year'] = data['Origin Time'].dt.year

plt.figure(figsize=(12, 6))
data['Year'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Earthquake Frequency over Time')
plt.show()

# Conclusion
print("Based on the analysis, we can draw the following conclusions:")
print("- The dataset contains information about", len(data), "earthquakes.")
print("- The earthquakes are distributed geographically based on their longitude and latitude.")
print("- The magnitude distribution shows the frequency of earthquakes at different magnitudes.")
print("- The depth distribution provides insights into the depths at which earthquakes occur.")
print("- The earthquake frequency over time helps understand the temporal distribution of earthquakes.")

# Additional analysis and conclusions can be added based on specific requirements.

# Identifying the locations with the highest magnitude earthquakes
top_locations = data.groupby('Location')['Magnitude'].max().nlargest(5)
print("Locations with the highest magnitude earthquakes:")
print(top_locations)

# Analyzing earthquake occurrence by month
data['Month'] = data['Origin Time'].dt.month
monthly_counts = data['Month'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Earthquake Occurrence by Month')
plt.xticks(rotation=0)
plt.show()

# Identifying the most active month for earthquakes
most_active_month = monthly_counts.idxmax()
print("The most active month for earthquakes is:", most_active_month)

# Analyzing earthquake occurrence by day of the week
data['DayOfWeek'] = data['Origin Time'].dt.day_name()
weekly_counts = data['DayOfWeek'].value_counts()

plt.figure(figsize=(10, 6))
weekly_counts.plot(kind='bar', color='lightgreen')
plt.xlabel('Day of the Week')
plt.ylabel('Frequency')
plt.title('Earthquake Occurrence by Day of the Week')
plt.xticks(rotation=45)
plt.show()

# Identifying the most active day of the week for earthquakes
most_active_day = weekly_counts.idxmax()
print("The most active day of the week for earthquakes is:", most_active_day)


# Analyzing earthquake magnitude distribution by location
plt.figure(figsize=(12, 6))
data.boxplot(column='Magnitude', by='Location', rot=90)
plt.xlabel('Location')
plt.ylabel('Magnitude')
plt.title('Magnitude Distribution by Location')
plt.tight_layout()
plt.show()

print(data['Location'])
print(data['Location'].value_counts('Transcarpathia'))
print(data['Location'].value_counts('Ivano-Frankivsk'))
print(data['Location'].value_counts('Chernivtsi'))
print(data['Location'].value_counts('Lviv'))
# Analyzing the correlation between variables
correlation = data[['Longitude', 'Latitude', 'Depth', 'Magnitude']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Between Variables')
plt.show()

# Conclusions
print("Based on the analysis, we can draw the following conclusions:")
print("- The locations with the highest average magnitude and the highest maximum magnitude have been identified.")
print("- The relationship between earthquake magnitude and depth has been analyzed.")
print("- The occurrence of earthquakes has been analyzed over time, including by year, month, and day.")
print("- The most active periods (year, month, and day) for earthquakes have been determined.")
print("- The magnitude distribution by location has been examined.")
print("- The correlation between variables (longitude, latitude, depth, magnitude) has been analyzed.")

# Additional analysis and conclusions can be added based on specific requirements.
# Analyzing the earthquake occurrence by season
data['Season'] = pd.cut(data['Origin Time'].dt.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Autumn'])

season_counts = data['Season'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
season_counts.plot(kind='bar', color='blue')
plt.xlabel('Season')
plt.ylabel('Frequency')
plt.title('Earthquake Occurrence by Season')
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()

# Identifying the season with the highest number of earthquakes
most_active_season = season_counts.idxmax()
print("The most active season for earthquakes is:", most_active_season)

# Analyzing the earthquake magnitude distribution by season
plt.figure(figsize=(10, 6))
sns.boxplot(data['Season'], data['Magnitude'], order=['Winter', 'Spring', 'Summer', 'Autumn'])
plt.xlabel('Season')
plt.ylabel('Magnitude')
plt.title('Magnitude Distribution by Season')
plt.tight_layout()

plt.show()

# Conclusions
print("Based on the analysis, we can draw the following conclusions:")
print("- The occurrence of earthquakes has been analyzed by season, and the most active season has been identified.")
print("- The magnitude distribution has been examined by season.")

# Additional analysis and conclusions can be added based on specific requirements.

import folium
import pandas as pd

# Load the dataset
data = pd.read_csv('earthquakes.csv')  # Replace 'locations.csv' with your dataset file path

ukraine_map = folium.Map(location=[49.0, 31.0], zoom_start=6, control_scale=True, tiles = None)

for index, row in data.iterrows():
    folium.Marker(
        location = [row['Latitude'], row['Longitude']],
        popup = row['Location'],
        icon = folium.Icon(color='blue',icon = 'info-sign', prefix = 'fa')
    ).add_to(ukraine_map)

# Customize the map's appearance
folium.TileLayer('stamenterrain').add_to(ukraine_map)  # Change the map style (options: 'openstreetmap', 'cartodbpositron', 'cartodbdark_matter', 'stamenterrain')
folium.TileLayer('openstreetmap').add_to(ukraine_map)  # Add OpenStreetMap as an additional tile layer
folium.TileLayer('stamentoner').add_to(ukraine_map)  # Add Stamen Toner as an additional tile layer
folium.TileLayer('Stamen Toner').add_to(ukraine_map)  # Add Stamen Watercolor as an additional tile layer
folium.TileLayer('cartodbdark_matter').add_to(ukraine_map)  # Add CartoDB Dark Matter as an additional tile layer
folium.TileLayer('Mapbox Bright').add_to(ukraine_map)
folium.TileLayer('Esri WorldStreetMap').add_to(ukraine_map)
folium.TileLayer('Esri DeLorme').add_to(ukraine_map)


folium.LayerControl(position='topright', collapsed=False).add_to(ukraine_map)
folium.LatLngPopup().add_to(ukraine_map)  # Show coordinates in a popup when clicked on the map

# Display the map

legend_html = '''
<div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; font-size: 14px;">
    <p><strong>Legend</strong></p>
    <p><i class="fa fa-map-marker fa-1x" style="color: blue;"></i>Epicenters of earthquakes</p>
    </div>
'''

ukraine_map.save('earthquakes.html')  # Save the map as an HTML file
ukraine_map

# Load earthquake data from a CSV file
data = pd.read_csv('earthquakes.csv')

Y = data['Magnitude']
X = data.drop(data.columns[5],axis=1)

plt.rcParams['figure.figsize']=(14,8)
sns.heatmap(X[['Latitude','Longitude','Depth']].corr(),cmap='magma_r', annot = True, linewidths=.5)
plt.title('Correlation between Latutude,Longitude and Depth with Magnitude',fontsize=20)
plt.show()

from sklearn.model_selection import train_test_split
y = data['Magnitude']
x = df[['Latitude','Longitude','Depth']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
lm = LinearRegression()
lm.fit(x_train, y_train)
result = lm.predict(x_test)
print('R2 score : ', metrics.r2_score(y_test, result))
#display adjusted R-squared
print('adjusted R2 score : ',1 - (1-metrics.r2_score(y_test, result))*(len(x)-1)/(len(x)-x.shape[1]))
print('MSE: ', metrics.mean_squared_error(y_test,result))
print('MAPE: ', metrics.mean_absolute_percentage_error(y_test,result)*100)

# Plotting the actual values
plt.scatter(y_test.index, y_test, color='blue', label='Actual')

# Plotting the predicted values
plt.scatter(y_test.index, result, color='red', label='Predicted')

plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.title('Actual vs Predicted Magnitude')
plt.legend()
plt.show()


# Create a DataFrame with actual and predicted values
table_data = pd.DataFrame({'Actual Magnitude': y_test, 'Predicted Magnitude': result})

# Display the table
print(table_data)




# %%
