import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the datasets
# Population data
population_data = pd.read_csv("/Users/raamizkhanniazi/Documents/5GDeployment/Data/population_data.csv")

# Tower data
tower_data = pd.read_csv("/Users/raamizkhanniazi/Documents/5GDeployment/Data/existing_towers.csv")

# Geographic data
geo_data = gpd.read_file("Data/geo_data.geojson")

# Step 2: Merge population data with geographic data
geo_data = geo_data.merge(population_data, on="region_id")

# Step 3: Calculate population density
geo_data["population_density"] = geo_data["population"] / geo_data["area"]

# Step 4: Calculate distance to the nearest existing tower
def calculate_min_distance(row, towers):
    point = row.geometry.centroid
    distances = [point.distance(Point(tower)) for tower in towers]
    return min(distances)

# Convert tower data to Point objects
towers = [Point(lon, lat) for lon, lat in zip(tower_data["longitude"], tower_data["latitude"])]
geo_data["distance_to_tower"] = geo_data.apply(lambda row: calculate_min_distance(row, towers), axis=1)

# Step 5: Normalize the features
scaler = MinMaxScaler()
geo_data[["population_density", "distance_to_tower"]] = scaler.fit_transform(
    geo_data[["population_density", "distance_to_tower"]]
)

# Step 6: Apply K-Means clustering to determine optimal tower locations
features = geo_data[["population_density", "distance_to_tower"]]
kmeans = KMeans(n_clusters=2, random_state=42)  # Adjust number of clusters as needed
geo_data["cluster"] = kmeans.fit_predict(features)

# Extract cluster centroids
centroids = kmeans.cluster_centers_

# Step 7: Visualize the results using Folium
# Initialize a map centered on the average coordinates of the regions
m = folium.Map(location=[geo_data.geometry.centroid.y.mean(), geo_data.geometry.centroid.x.mean()], zoom_start=13)

# Add regions to the map
for _, row in geo_data.iterrows():
    folium.GeoJson(
        row.geometry,
        tooltip=f"Region ID: {row['region_id']}<br>Cluster: {row['cluster']}"
    ).add_to(m)

# Add existing towers to the map
for tower in towers:
    folium.Marker(
        location=[tower.y, tower.x],
        popup="Existing Tower",
        icon=folium.Icon(color="blue")
    ).add_to(m)

# Add proposed tower locations (cluster centroids)
for centroid in centroids:
    folium.Marker(
        location=[centroid[1], centroid[0]],  # Latitude, Longitude
        popup="Proposed Tower Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

# Save the map
m.save("5g_deployment_analysis.html")
print("Map saved as 5g_deployment_analysis.html")
