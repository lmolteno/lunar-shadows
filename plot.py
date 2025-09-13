import pickle
import sys
from time import sleep
import matplotlib.patches as mpatches
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import numpy as np
import geopandas as gpd
import pyproj
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from tqdm import tqdm

nz_mercator = ccrs.epsg(2193)

# plt.ion()
ax = plt.axes(projection=nz_mercator)
plt.gcf().set_size_inches(12, 11)
# ax.gridlines()
plt.gcf().set_layout_engine('tight')

water_color = '#0F0F0F'
road_color = '#0F0F2F'

ax.set_facecolor(water_color)

coastlines = gpd.read_file("map-data/nz-coastlines-and-islands-polygons-topo-1500k/nz-coastlines-and-islands-polygons-topo-1500k.gpkg")
contours = gpd.read_file("map-data/nz-contours-topo-1500k/nz-contours-topo-1500k.gpkg")
lakes = gpd.read_file("map-data/nz-lake-polygons-topo-1500k/nz-lake-polygons-topo-1500k.gpkg")
roads = gpd.read_file("map-data/nz-road-centrelines-topo-1500k/nz-road-centrelines-topo-1500k.gpkg")
rivers = gpd.read_file("map-data/nz-river-centrelines-topo-1500k/nz-river-centrelines-topo-1500k.gpkg")
shingles = gpd.read_file("map-data/nz-shingle-polygons-topo-1500k/nz-shingle-polygons-topo-1500k.gpkg")

ax.add_geometries(coastlines.geometry, crs=nz_mercator, facecolor='white', edgecolor='none')
# ax.add_geometries(snow.geometry, crs=nz_mercator, facecolor='white', edgecolor='black', linewidth=0.2)
ax.add_geometries(rivers.geometry, crs=nz_mercator, fc='none', edgecolor=water_color, alpha=0.5, linewidth=0.3)
ax.add_geometries(shingles.geometry, crs=nz_mercator, fc=water_color, edgecolor='none', alpha=0.2)
ax.add_geometries(contours.geometry, crs=nz_mercator, facecolor='none', edgecolor='black', alpha=0.1)
ax.add_geometries(lakes.geometry, crs=nz_mercator, facecolor=water_color, edgecolor='none')
ax.add_geometries(roads.geometry, crs=nz_mercator, facecolor='none', edgecolor=road_color, alpha=0.5, linewidth=0.3)
# highways
ax.add_geometries(roads[roads.hway_num.notna()].geometry, crs=nz_mercator, facecolor='none', edgecolor=road_color, alpha=0.5, linewidth=1)

with open("corrected_antares.pickle", "rb") as f:
    antares_edges = pickle.load(f)

with open("corrected_antares_b.pickle", "rb") as f:
    antares_b_edges = pickle.load(f)
    print(antares_b_edges)

antares_patch = mpatches.Polygon([[0, 0]], closed=True, fill=True, fc="tab:red", ec='none', alpha=0.7, transform=ccrs.Geodetic())
antares_b_patch = mpatches.Polygon([[0, 0]], closed=True, fill=True, fc="tab:cyan", ec='none', alpha=0.3, transform=ccrs.Geodetic())
ax.add_patch(antares_patch)
ax.add_patch(antares_b_patch)

geod = pyproj.Geod(ellps="WGS84")

def process(lonlats):
    lons, lats, _ = np.array(lonlats).T
    lons = lons[::-1]
    lats = lats[::-1]

    # Create arrays of start and end points, with wrap-around
    start_lons = lons
    start_lats = lats
    end_lons = np.roll(lons, -1)  # Shift array to get next point
    end_lats = np.roll(lats, -1)  # Shift array to get next point

    _, _, distances = geod.inv(start_lons, start_lats, end_lons, end_lats)

    mask = distances <= 50000

    if distances[-1] > 50000:
        mask[-1] = False

    # Apply mask to filter points
    filtered_lons = np.asarray(lons)[mask]
    filtered_lats = np.asarray(lats)[mask]

    return (filtered_lons, filtered_lats)

logo = plt.imread("./das-logo.png")
imagebox = OffsetImage(logo, zoom = 0.5)
ab = AnnotationBbox(imagebox, (0.95, 0.05), frameon = True, xycoords='axes fraction', box_alignment=(1, 0))
ab.patch.set(fc=water_color)
ax.add_artist(ab)
ab.set_zorder(11)

for time in tqdm(sorted(antares_edges.keys())[100:]):

    antares = antares_edges[time]
    antares_b = antares_b_edges[time]

    if len(antares[0]) == 0 or len(antares_b[0]) == 0:
        continue

    lon, lat = process(antares)
    lonb, latb = process(antares_b)

    antares_patch.set_xy(np.asarray([lon, lat]).T)
    antares_b_patch.set_xy(np.asarray([lonb, latb]).T)
    antares_patch.set_zorder(9)
    antares_b_patch.set_zorder(10)
    plt.title(f"Lunar Occultation of Antares A and B\n$\\mathtt{{{time}Z}}$")

    #ax.set_extent([169, 170.8, -43.5, -44.8], crs=ccrs.PlateCarree())
    ax.set_extent([165, 175, -40, -47], crs=ccrs.PlateCarree())

    # plt.gcf().canvas.draw()
    # plt.gcf().canvas.flush_events()

    # if sys.argv[1] == '--show':
    #     plt.show()
    #     exit()
    # else:
    plt.savefig(f'pics/{time}.png')

    # input()
    # exit()

# plt.show()
