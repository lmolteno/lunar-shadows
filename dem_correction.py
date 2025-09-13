import pickle
import time
import rasterio
from rasterio.sample import sample_gen
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, ITRS, EarthLocation
from astropy.time import Time
import astropy.units as u
from pyproj import Transformer
from shapely.geometry.linestring import LineString
from tqdm import tqdm
import geopandas as gpd


def ecef_to_latlon_batch(x, y, z):
    """
    Convert ECEF coordinates to latitude, longitude, and height on WGS84.

    Parameters:
    ----------
    x, y, z : array-like
        ECEF X, Y, Z coordinates in meters

    Returns:
    -------
    lat, lon, height : numpy arrays
        Latitude and longitude in degrees, height in meters
    """
    transformer = Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True
    )

    lon, lat, height = transformer.transform(x, y, z)

    return lon, lat, height


def sample_location(itrs, dem_dataset, dem_data):
    lon, lat, ellipsoidal_height = EarthLocation.from_geocentric(*itrs, unit=u.m).geodetic
    if contains(dem_dataset, lon.value, lat.value):
        row, col = dem_dataset.index(lon.value, lat.value)
        return dem_data[row, col], ellipsoidal_height.to(u.m).value
    else:
        return 0, ellipsoidal_height.to(u.m).value

def contains(dem_data, lon, lat):
    return dem_data.bounds.left < lon < dem_data.bounds.right and dem_data.bounds.bottom < lat < dem_data.bounds.top


step_size = 10 # metres
coastlines_dataset = gpd.read_file("map-data/nz-coastlines-and-islands-polygons-topo-1500k/nz-coastlines-and-islands-polygons-topo-1500k.gpkg")
coastlines = coastlines_dataset[coastlines_dataset['name'] == "South Island or Te Waipounamu"].geometry.to_crs(epsg=4326)


def correct_point(obstime, star_vec_normed, lon, lat, dem_dataset: rasterio.DatasetReader, dem_data):
    """ This corrects an ellipsoidal intersection point, projecting it along the star's sight path to find the intersection with the terrain."""

    if not contains(dem_dataset, lon, lat):
        return np.array([lon, lat, 0])

    start_loc = np.array([*map(lambda f: f.value, EarthLocation.from_geodetic(lon * u.degree, lat * u.degree).geocentric)])
    max_height = 4000 # highest point in nz above ellipsoid
    ellipsoidal_height = 0
    terrain_height = 0
    dist = step_size
    current_point = start_loc
    under_terrain = True
    intersection_points = []
    ecef_points = start_loc + np.outer(np.linspace(0, 226000, 2), star_vec_normed)
    lons, lats, ray_heights = ecef_to_latlon_batch(ecef_points[:, 0], ecef_points[:, 1], ecef_points[:, 2])
    sample_line = LineString(np.array([lons, lats]).T)
    if not coastlines.iloc[0].envelope.intersects(sample_line):
        return np.array([lon, lat, 0])
    if not coastlines.intersects(sample_line).any():
        return np.array([lon, lat, 0])
    start = time.perf_counter_ns()
    ecef_points = start_loc + np.outer(np.arange(0, 226000, 90), star_vec_normed)
    lons, lats, ray_heights = ecef_to_latlon_batch(ecef_points[:, 0], ecef_points[:, 1], ecef_points[:, 2])
    condition = ray_heights < 4000
    lons, lats, ray_heights = lons[condition], lats[condition], ray_heights[condition]
    # result = np.array([*sample_gen(dem_dataset, np.array([lons, lats]).T, masked=True)])
    terrain_heights = np.array([*dem_dataset.sample(np.array([lons, lats]).T, indexes=1, masked=True)])[:, 0]
    difference = ray_heights - terrain_heights
    crossings = np.where((difference[:-1] < -0.5) & (difference[1:] > 0))[0] + 1
    if len(crossings) == 0:
        return np.array([lon, lat, 0])

    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    # ax.plot(lons, lats)
    # ax.set_extent([165, 177, -40, -48], crs=ccrs.PlateCarree())
    # plt.show()
    # print(f"Sampled {len(result)} and found a max height of {np.max(result)}")

    # return lon, lat, 0
    # plt.plot(terrain_heights)
    # plt.plot(ray_heights)
    # for crossing in crossings:
    #     plt.axvline(crossing)
    # plt.show()
    crossing_to_use = crossings[-1]
    return np.array([lons[crossing_to_use], lats[crossing_to_use], terrain_heights[crossing_to_use]])


def main():
    dem_file = "copernicus/rasters_COP90/output_hh.tif"
    dem_dataset = rasterio.open(dem_file)
    dem_data = dem_dataset.read(1)
    try:
        star_coords = {
                "antares": SkyCoord.from_name("Antares"),
                "antares_b": SkyCoord.from_name("Antares").directional_offset_by(277 * u.deg, 2.8 * u.arcsec)
        }
        for file in ["antares_edges.pickle", "antares_b_edges.pickle"]:
            star_name = file.split("_edges")[0]
            with open(file, 'rb') as f:
                edges = pickle.load(f)

            corrected_dict = {}

            for time_str in tqdm(sorted(edges.keys())):
                corrected_points = []
                obs_time = Time(time_str)
                star = star_coords[star_name].transform_to(ITRS(obstime=obs_time))
                star_normed = star.cartesian.xyz / np.linalg.norm(star.cartesian.xyz)
                coords = edges[time_str].T
                for lon, lat in tqdm(coords, leave=False):
                    corrected_point = correct_point(obs_time, star_normed.value, lon, lat, dem_dataset, dem_data)
                    corrected_points.append(corrected_point)
                corrected_dict[time_str] = corrected_points

            with open(f"corrected_{star_name}.pickle", "wb") as f:
                pickle.dump(corrected_dict, f)
    finally:
        dem_dataset.close()

if __name__ == "__main__":
    main()
