from astropy import units as u
import cartopy.crs as ccrs
from shapely import Polygon
import numpy as np
import pickle
import pandas as pd
import rasterio
from astropy.coordinates import SkyCoord, EarthLocation, GCRS, AltAz, get_body, concatenate, Angle
from astropy.time import Time
from astropy.timeseries import TimeSeries
from lunarsky import MCMF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from shapely.geometry.multipolygon import MultiPolygon
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from main import MOON_RADIUS
from utils import generate_spherical_gridlines_to_3d_array

TESTING=False

if TESTING:
    dem_data = np.zeros((46080, 92160))
else:
    with rasterio.open("./Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif") as src:
        print("Reading file...")
        dem_data = src.read(1)
        print("Read")



def sample_vector(limb_vecs):
    lon = np.degrees(np.arctan2(limb_vecs[1, :], limb_vecs[0, :]))
    lat = np.degrees(np.arcsin(limb_vecs[2, :]))

    px = np.array(256 * (lon + 180), dtype=int)
    py = np.array(256 * (lat + 90), dtype=int)

    py = np.minimum(py, 46080 - 1)
    px = np.where(px >= 92160, px - 92160, px)

    elevation = dem_data[py, px] * u.m

    return elevation


def plot_sphere(ax, radius, offset=None):
    if offset is None:
        offset = [0, 0, 0]

    u_1, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u_1) * np.sin(v)
    y = np.sin(u_1) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe((x * radius) + offset[0], (y * radius) + offset[1], (z * radius) + offset[2], color="r", zorder=-1, alpha=0.1)


def lunar_horizon(observer: EarthLocation | SkyCoord, obs_time: Time, resolution):
    if not isinstance(observer, SkyCoord):
        observer_mcmf = observer.get_itrs(obs_time).transform_to(MCMF(obstime=obs_time))
    else:
        observer_mcmf = observer.transform_to(MCMF(obstime=obs_time))
    observer_mcmf_unit = (observer_mcmf.cartesian.xyz / np.linalg.norm(observer_mcmf.cartesian.xyz)).value.T[0]
    observer_distance = np.linalg.norm(observer_mcmf.cartesian.xyz)

    azimuths = np.linspace(0, 360, resolution, endpoint=False) * u.deg

    limb_circle_distance = ((MOON_RADIUS * MOON_RADIUS) / observer_distance).to(u.m)
    limb_circle_radius = ((MOON_RADIUS * np.sqrt(observer_distance ** 2 - MOON_RADIUS ** 2)) / observer_distance).to(u.m)

    # now we want to find points in this plane.
    limb_circle_center = observer_mcmf_unit * limb_circle_distance

    # Create a basis in the plane perpendicular to the observer vector
    # We need two orthogonal vectors u and v that are also orthogonal to observer_unit_vector
    # A simple way to find one such vector is to take the cross product with a non-parallel vector.
    # Avoid the case where observer_unit_vector is aligned with the z-axis.
    if not np.allclose(np.abs(observer_mcmf_unit), [0, 0, 1]):
        temp_vector = np.array([0, 0, 1])
    else:
        temp_vector = np.array([1, 0, 0])

    u_vec = np.cross(observer_mcmf_unit, temp_vector)
    u_unit = u_vec / np.linalg.norm(u_vec)
    v_unit = np.cross(observer_mcmf_unit, u_unit).reshape(-1, 1)
    u_unit = u_unit.reshape(-1, 1)

    # Calculate the points on the limb
    sphere_points = limb_circle_center.reshape(-1, 1) + (limb_circle_radius.reshape(-1, 1) * (u_unit * np.cos(azimuths) + v_unit * np.sin(azimuths)))

        # surface_coord = SkyCoord(MCMF(sphere_point, obstime=obs_time))# .transform_to(GCRS(obstime=obs_time))
        #
        # print(surface_coord.cartesian.xyz)
        # print(np.linalg.norm(surface_coord.cartesian.xyz))
    # ax.plot(*(observer_mcmf_unit * observer_distance).to(u.m), 'ro')
    # ax.quiver(0, 0, 0, *(observer_mcmf_unit * 400 * u.km).to(u.m))

    # for each sphere_point, the direction to the observer
    observer_directions = (observer_mcmf_unit * observer_distance).reshape(-1, 1) - sphere_points
    observer_directions = observer_directions / np.linalg.norm(observer_directions, axis=0)

    observer_directions = observer_directions[:, :, np.newaxis]
    sphere_points = sphere_points[:, :, np.newaxis]
    distances = (np.arange(start=-400, stop=400, step=0.1) * u.km).reshape(1, 1, -1)
    sample_points = sphere_points + observer_directions * distances
    sample_points_unit = sample_points / np.linalg.norm(sample_points, axis=0)
    sampled_points = sample_points_unit * (sample_vector(sample_points_unit.value) + MOON_RADIUS)

    sphere_norms = (sphere_points / np.linalg.norm(sphere_points, axis=0))

    scalar_projections = np.sum(sampled_points * sphere_norms, axis=0) * sphere_norms
    projection_norms = np.linalg.norm(scalar_projections, axis=0)
    highest_indices = np.argmax(projection_norms, axis=1)

    row_indices = np.arange(resolution)
    best_points = scalar_projections[:, row_indices, highest_indices]

    return SkyCoord(MCMF(best_points, obstime=obs_time)).transform_to(GCRS(obstime=obs_time))

def generate_polygons(crater_lats, crater_lons, crater_diameters, n_points=5):
    n_craters = len(crater_lats)
    lats = crater_lats
    lons = crater_lons

    centers_x = MOON_RADIUS * np.cos(lats) * np.cos(lons)
    centers_y = MOON_RADIUS * np.cos(lats) * np.sin(lons)
    centers_z = MOON_RADIUS * np.sin(lats)
    centers = np.column_stack((centers_x, centers_y, centers_z))

    # Normalize centers to get unit vectors
    # Shape: (n_craters, 3)
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    center_units = centers / center_norms

    # Create first tangent vectors for all craters at once
    # For most craters, use a vector perpendicular to center and z-axis
    # Shape: (n_craters, 3)
    tangent1 = np.zeros((n_craters, 3))

    # For craters not near poles (most craters)
    not_near_poles = np.abs(center_units[:, 2]) < 0.9
    if np.any(not_near_poles):
        tangent1[not_near_poles, 0] = -center_units[not_near_poles, 1]
        tangent1[not_near_poles, 1] = center_units[not_near_poles, 0]
        tangent1[not_near_poles, 2] = 0

    # For craters near poles
    near_poles = ~not_near_poles
    if np.any(near_poles):
        tangent1[near_poles, 0] = 1
        tangent1[near_poles, 1] = 0
        tangent1[near_poles, 2] = 0

    # Normalize the first tangent vectors
    # Shape: (n_craters, 3)
    tangent1_norms = np.linalg.norm(tangent1, axis=1, keepdims=True)
    tangent1 = tangent1 / tangent1_norms

    # Create second tangent vectors using cross product
    # Shape: (n_craters, 3)
    tangent2 = np.cross(center_units, tangent1)

    # Normalize the second tangent vectors
    # Shape: (n_craters, 3)
    tangent2_norms = np.linalg.norm(tangent2, axis=1, keepdims=True)
    tangent2 = tangent2 / tangent2_norms

    # Calculate angular radii for all craters
    # Shape: (n_craters,)
    radii_km = crater_diameters / 2
    angular_radii = np.arcsin(np.clip(radii_km / MOON_RADIUS, 0, 1))

    # Generate evenly spaced angles for all craters at once
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Expand dimensions to allow broadcasting
    center_units_expanded = center_units[:, np.newaxis, :]  # (n_craters, 1, 3)
    tangent1_expanded = tangent1[:, np.newaxis, :]      # (n_craters, 1, 3)
    tangent2_expanded = tangent2[:, np.newaxis, :]      # (n_craters, 1, 3)
    angular_radii_expanded = angular_radii[:, np.newaxis, np.newaxis]  # (n_craters, 1, 1)
    cos_theta_expanded = cos_theta[np.newaxis, :, np.newaxis]      # (1, n_points, 1)
    sin_theta_expanded = sin_theta[np.newaxis, :, np.newaxis]      # (1, n_points, 1)

    # Calculate rim points using vectorized operations
    # Shape: (n_craters, n_points, 3)
    rim_points = (center_units_expanded * np.cos(angular_radii_expanded)) + \
                 (tangent1_expanded * np.sin(angular_radii_expanded) * cos_theta_expanded) + \
                 (tangent2_expanded * np.sin(angular_radii_expanded) * sin_theta_expanded)

    # Scale to Moon radius
    all_polygons = rim_points * MOON_RADIUS
    return all_polygons


def project_craters(crater_polygons, observer: EarthLocation, obstime: Time):
    observer_mcmf = observer.get_itrs(obstime).transform_to(MCMF(obstime=obstime))
    north_dir = np.dot(crater_polygons, np.array([[0], [0], [MOON_RADIUS.to_value(u.km)]]) * u.km)
    obs_dir = np.dot(crater_polygons, observer_mcmf.cartesian.xyz)
    mask = np.all(obs_dir > 0, axis=1) & np.all(north_dir.value * u.km > (MOON_RADIUS / 2), axis=1)
    n_points = crater_polygons.shape[1]
    flattened_polygons = crater_polygons[np.repeat(mask, n_points, 1)]
    # reprojected_polygons = MCMF(*flattened_polygons.T, obstime=obstime).transform_to(GCRS(obstime=obstime)).cartesian.xyz.reshape(-1, n_points, 3)
    reprojected_polygons = MCMF(*flattened_polygons.T, obstime=obstime).transform_to(AltAz(obstime=obstime, location=observer))
    lons = reprojected_polygons.spherical.lon.reshape(-1, n_points)
    lats = reprojected_polygons.spherical.lat.reshape(-1, n_points)
    return np.stack((lons, lats), axis=2)

def get_projected_gridlines(observer: EarthLocation, obstime: Time):
    observer_mcmf = observer.get_itrs(obstime).transform_to(MCMF(obstime=obstime))
    coords = generate_spherical_gridlines_to_3d_array(resolution=50 if TESTING else 360)
    obs_dir = np.dot(coords, observer_mcmf.cartesian.xyz / np.linalg.norm(observer_mcmf.cartesian.xyz))
    flattened_coords = coords[np.repeat(obs_dir > 0, 3, 2)]

    # ax = plt.subplot(projection='3d')
    reshaped = flattened_coords.reshape((-1, 3))
    # elevations = sample_vector(reshaped.T)
    distances = np.sqrt(np.sum(np.diff(reshaped, axis=0)**2, axis=1))
    consequent = distances < 0.2

    lines = []
    current_line_points = []
    for i in range(len(consequent)):
        if consequent[i]:
            # If current_line_points is empty, it's the start of a new segment
            if not current_line_points:
                current_line_points.append(reshaped[i] * (MOON_RADIUS).to(u.m))
            current_line_points.append(reshaped[i+1] * (MOON_RADIUS).to(u.m))
        else:
            # If there was a current_line and the condition is now false,
            # or if it's the end of the loop and there's a current_line
            if current_line_points:
                lines.append(np.array(current_line_points))
                current_line_points = [] # Reset for the next potential line

    # Check if the last segment was being built when the loop ended
    if current_line_points:
        lines.append(np.array(current_line_points))

    alt_az_lines = []

    for line in lines:
        altaz = SkyCoord(MCMF(*(line * u.m).T, obstime=obstime)).transform_to(AltAz(obstime=obstime, location=observer))
        alt_az_lines.append(altaz)

    return alt_az_lines

def get_terminator(observer: EarthLocation, obstime: Time, observer_horizon: SkyCoord, resolution):
    sun_mcmf = get_body('sun', time=obstime).transform_to(MCMF(obstime=obstime))
    sun_horizon = lunar_horizon(sun_mcmf, obstime, resolution=resolution)
    moon = get_body('moon', time=obstime) # gcrs
    mask = sun_horizon.distance < moon.distance
    i_end = np.nonzero(mask == False)[0][0]
    i_start = np.nonzero(mask == True)[0][i_end]
    end_coord = sun_horizon[i_end]
    start_coord = sun_horizon[i_start]
    matching_end_index, _, _ = end_coord.match_to_catalog_3d(observer_horizon)
    matching_start_index, _, _ = start_coord.match_to_catalog_3d(observer_horizon)
    observer_horizon_section = np.take(observer_horizon, np.arange(matching_end_index, matching_start_index - len(observer_horizon), -1))

    visible_terminator = concatenate([sun_horizon[:i_end], observer_horizon_section, sun_horizon[i_start:]])
    return visible_terminator

def main():
    timeseries = TimeSeries(time_start='2025-08-31T11:47:00Z', time_delta=10 * u.second, n_samples=6 * 20)
    time_points = [row['time'] for row in timeseries]
    # observer = EarthLocation.of_site('MJO')
    # observer_name = 'UCMJO'

    # observer = EarthLocation.from_geodetic(Angle('169d10m10s'), Angle('-44d21m07s'), height=286.69 * u.m)
    # observer_name = 'BoundaryCreekCampsite'

    # observer = EarthLocation.from_geodetic(Angle('169d10m42s'), Angle('-44d22m06s'), height=289.25 * u.m)
    # observer_name = 'SheepskinCreek'

    observer = EarthLocation.from_geodetic(Angle('170d11m26s'), Angle('-43d55m04s'), height=541.686 * u.m)
    observer_name = 'NorthEastPukaki'

    from pathlib import Path
    Path(observer_name).mkdir(exist_ok=True)

    logo = plt.imread("./das-logo.png")
    water_color = '#0F0F0F'

    out_dict = {}

    crater_df = pd.read_csv('map-data/moon/lunar_crater_database_robbins_2018_bundle/data/big_craters.csv')
    crater_df = crater_df[crater_df['DIAM_CIRC_IMG'] > 20]
    crater_polygons = generate_polygons(crater_df['LAT_CIRC_IMG'].values * u.deg, crater_df['LON_CIRC_IMG'].values * u.deg, crater_df['DIAM_CIRC_IMG'].values * u.km, 15)

    for obstime in tqdm(time_points):
        horizon = lunar_horizon(observer, obstime, 36 if TESTING else 360 * 10)
        terminator = get_terminator(observer, obstime, horizon, 36 if TESTING else 360 * 10)
        out_dict[obstime] = horizon.cartesian.xyz
        # ax.scatter(*sampled_points, c=elevations, cmap=plt.cm.jet)

        local_altaz = AltAz(obstime=obstime, location=observer)

        moon = get_body('moon', obstime, observer).transform_to(local_altaz)
        horizon_altaz = horizon.transform_to(local_altaz)
        terminator_altaz = terminator.transform_to(local_altaz)

        antares_a = SkyCoord.from_name('Antares').transform_to(local_altaz)
        antares_b = SkyCoord.from_name('Antares B').transform_to(local_altaz)

        plt.clf()
        plt.gcf().set_size_inches(10, 10)
        plt.gcf().set_layout_engine('tight')
        ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=antares_a.az.to_value(u.deg), central_latitude=antares_a.alt.to_value(u.deg)))
        ax.set_facecolor(water_color)
        imagebox = OffsetImage(logo, zoom=0.5)
        ab = AnnotationBbox(imagebox, (0.95, 0.05), frameon=False, xycoords='axes fraction', box_alignment=(1, 0))
        ab.set_zorder(11)
        ax.add_artist(ab)
        # ax.plot(, 'k-', transform=ccrs.Geodetic())
        horizon_patch = mpatches.Polygon(np.array([horizon_altaz.az.to_value(u.deg), horizon_altaz.alt.to_value(u.deg)]).T, fc='white', ec='none', alpha=0.5, transform=ccrs.PlateCarree())
        ax.add_patch(horizon_patch)
        terminator_patch= mpatches.Polygon(np.array([terminator_altaz.az.to_value(u.deg), terminator_altaz.alt.to_value(u.deg)]).T[::-1], fc='black', ec='none', alpha=0.5, transform=ccrs.PlateCarree())
        ax.add_patch(terminator_patch)
        ax.scatter(antares_b.az.to_value(u.deg), antares_b.alt.to_value(u.deg), s=3, transform=ccrs.Geodetic())
        ax.scatter(antares_a.az.to_value(u.deg), antares_a.alt.to_value(u.deg), s=3, transform=ccrs.Geodetic())

        print("Projecting craters")
        projected_craters = project_craters(crater_polygons, observer, obstime)
        tuple_craters = [
            [[tuple(coord) for coord in polygon]]
            for polygon in projected_craters.to_value(u.deg)
        ]
        craters_patch = MultiPolygon(tuple_craters)
        print("Projected craters")
        ax.add_geometries(craters_patch, fc=water_color, ec='none', alpha=0.3, crs=ccrs.PlateCarree())

        print("Projecting gridlines")
        gridlines = get_projected_gridlines(observer, obstime)
        print("Projected craters")

        for line in gridlines:
            ax.plot(line.az.to_value(u.deg), line.alt.to_value(u.deg), transform=ccrs.PlateCarree(), color='black', linewidth=0.5)

        plot_radius = 0.05

        ax.set_extent([
            antares_a.az.to_value(u.deg) - plot_radius,
            antares_a.az.to_value(u.deg) + plot_radius,
            antares_a.alt.to_value(u.deg) - plot_radius,
            antares_a.alt.to_value(u.deg) + plot_radius
        ])

        ax.set_title(f"Lunar Occultation of Antares A and B from {observer_name}\n$\\mathtt{{{obstime}Z}}$")
        ax.set_aspect('equal')
        if TESTING:
            plt.show()
        else:
            plt.savefig(f'{observer_name}/{obstime}.png')

    with open(f'{observer_name}_horizons.pickle', 'wb') as f:
        pickle.dump(out_dict, f)

if __name__ == '__main__':
    main()
