from time import perf_counter_ns

import matplotlib.pyplot as plt
import pickle
from cartopy.crs import WGS84_SEMIMAJOR_AXIS, WGS84_SEMIMINOR_AXIS
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rasterio
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.coordinates import get_body, solar_system_ephemeris
from astropy.table import Table, Column
from lunarsky import SkyCoord, MoonLocation, Time, MCMF
from astropy.timeseries import TimeSeries
import cartopy.crs as ccrs
from tqdm import tqdm


MOON_RADIUS=1737.4 * u.km
PLOT=False


def plot_sphere(ax, radius, offset=[0,0,0]):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe((x * radius) + offset[0], (y * radius) + offset[1], (z * radius) + offset[2], color="r")

def sample_vector(dem_data, limb_vecs):
    lon = np.degrees(np.arctan2(limb_vecs[:, 1], limb_vecs[:, 0]))
    lat = np.degrees(np.arcsin(limb_vecs[:, 2]))

    # lon = np.degrees(np.arctan2(limb_vec[1], limb_vec[0]))
    # lat = np.degrees(np.arcsin(limb_vec[2]))

    # Look up elevation from DEM at this point
    # Convert lat/lon to pixel coordinates for the DEM
    # px, py = int(256 * (lon + 180)), int(256 * (lat + 90))
    px = np.array(256 * (lon + 180), dtype=int)
    py = np.array(256 * (lat + 90), dtype=int)

    # Lookup elevation at this point
    elevation = dem_data[py, px] * u.m

    return elevation


def calculate_lunar_horizon(star_coord, obs_time, dem_data, resolution=360):
    """
    Calculate the lunar horizon profile as seen from Earth for a specific star.

    Parameters:
    -----------
    star_coord : SkyCoord
        Coordinates of the star in ICRS frame
    obs_time : Time
        Observation time
    dem_file : str, optional
        Path to lunar DEM (Digital Elevation Model) TIFF file
    resolution : int
        Number of points to sample around the lunar limb (default: 360, i.e., 1 degree steps)

    Returns:
    --------
    azimuths : array
        Azimuths around the lunar limb (in degrees)
    elevations : array
        Maximum elevations (heights above lunar reference radius) at each azimuth
    """
    # Convert star coordinates to MCMF frame
    star_mcmf = star_coord.transform_to(MCMF(obstime=obs_time))

    # Define azimuths around the lunar limb
    azimuths = np.linspace(0, 360, resolution, endpoint=False) * u.deg

    # Initialize array to store maximum elevations
    elevations = np.zeros(len(azimuths)) * u.km

    shadow_locations = []
    timings = []

    # For each azimuth, find the point on the lunar limb
    for i, az in tqdm([*enumerate(azimuths)]):
        azi_start = perf_counter_ns()

        # Calculate the direction from the lunar center toward the star
        # but perpendicular to the star direction at the given azimuth

        # Extract the star direction vector
        star_vec = np.array([star_mcmf.cartesian.x.value[0],
                             star_mcmf.cartesian.y.value[0],
                             star_mcmf.cartesian.z.value[0]])
        star_vec = star_vec / np.linalg.norm(star_vec)

        # Create a reference vector perpendicular to star_vec
        # This is arbitrary but will be used to define the limb plane
        if np.abs(star_vec[2]) < 0.9:
            ref_vec = np.array([0, 0, 1])
        else:
            ref_vec = np.array([1, 0, 0])

        # Find perpendicular vector to star_vec in the plane
        perp1 = np.cross(star_vec, ref_vec)
        perp1 = perp1 / np.linalg.norm(perp1)

        # Find another perpendicular vector to create a basis in the plane
        perp2 = np.cross(star_vec, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Calculate limb vector at the given azimuth in the plane perpendicular to star_vec
        limb_vec = perp1 * np.cos(az.to_value(u.rad)) + perp2 * np.sin(az.to_value(u.rad))

        # Find the corresponding point on the limb
        # The limb is where this vector intersects the lunar surface
        if True:
            elevation = sample_vector(dem_data, np.array([limb_vec]))

            surface_coord = (limb_vec * (elevation + MOON_RADIUS))

            original_vec_time = perf_counter_ns()

            distances = (np.arange(start=-10, stop=10, step=0.1) * u.km).to(u.m).value.reshape(-1, 1)
            translated_coords = surface_coord.value + (star_vec * distances)
            unit_translated_coord = translated_coords / np.linalg.norm(translated_coords, axis=1).reshape(-1, 1)
            new_elevation = sample_vector(dem_data, unit_translated_coord).reshape(-1, 1)
            new_surface_coord = (unit_translated_coord * (new_elevation + MOON_RADIUS))

            scalar_projections = (new_surface_coord * limb_vec).sum(axis=1).reshape(-1, 1) * limb_vec
            projection_norms = np.linalg.norm(scalar_projections, axis=1)
            max_projection_idx = np.argmax(projection_norms)

            if projection_norms[max_projection_idx] > np.linalg.norm(surface_coord):
                # print(f"Found point with greater norm - {projection_norms[max_projection_idx]} rather than {np.linalg.norm(surface_coord)}")
                surface_coord = scalar_projections[max_projection_idx]

            max_vec_time = perf_counter_ns()


            surface_coord = SkyCoord(MCMF(surface_coord, obstime=obs_time)).transform_to(GCRS(obstime=obs_time))
            moon_point = surface_coord.cartesian.xyz.value.T[0]
            star_gcrs = star_coord.transform_to(GCRS(obstime=obs_time))
            star_point = star_gcrs.cartesian.xyz.value

            star_unit = star_point / np.linalg.norm(star_point)

            direction = -star_unit

            earth_radius = 6371000.0  # meters

            # Position is the Moon point relative to Earth center
            pos = moon_point

            discriminant = (np.dot(direction, moon_point) ** 2) - (np.linalg.norm(moon_point) ** 2 - earth_radius ** 2)

            if discriminant < 0:
                # print("Line does not intersect Earth")
                continue
                # raise ValueError("Line does not intersect Earth")

            b = np.dot(direction, moon_point)

            # Calculate solutions
            t1 = (-b - np.sqrt(discriminant))
            t2 = (-b + np.sqrt(discriminant))

            t = min(t1, t2)

            # Calculate the intersection point
            earth_intersection = pos + t * direction

            earth_gcrs = CartesianRepresentation(x=earth_intersection[0] * u.m,
                                                 y=earth_intersection[1] * u.m,
                                                 z=earth_intersection[2] * u.m)

            earth_gcrs = GCRS(earth_gcrs, obstime=obs_time)
            earth_itrs = earth_gcrs.transform_to(ITRS(obstime=obs_time))

            sphere_point_time = perf_counter_ns()

            # now iteratively intersect with the ellipsoid
            projected_ellipsoid = EarthLocation.from_geodetic(
                earth_itrs.earth_location.geodetic.lon,
                earth_itrs.earth_location.geodetic.lat,
                height=0 * u.m

            )

            # Calculate the ellipsoid normal at this point
            normal = np.array([projected_ellipsoid.x.value, projected_ellipsoid.y.value, projected_ellipsoid.z.value * (WGS84_SEMIMAJOR_AXIS / WGS84_SEMIMINOR_AXIS) ** 2])
            normal = normal / np.linalg.norm(normal)

            x, y, z = projected_ellipsoid.x.value, projected_ellipsoid.y.value, projected_ellipsoid.z.value

            # Iterative refinement (usually converges in 1-3 iterations)
            for _ in range(10):
                # Find the point on the ray that intersects the plane defined
                # by the current point and its normal
                t = np.dot(np.array([x,y,z]) - earth_itrs.cartesian.xyz.value, normal) / np.dot(direction, normal)
                intersection = earth_itrs.cartesian.xyz.value + t * direction

                # Project this intersection to the ellipsoid
                new_location = EarthLocation.from_geocentric(
                    intersection[0] * u.m,
                    intersection[1] * u.m,
                    intersection[2] * u.m
                )

                # Get the surface point (height=0)
                new_location= EarthLocation.from_geodetic(
                    new_location.lon,
                    new_location.lat,
                    height=0 * u.m
                )

                # Check if we've converged
                x_new, y_new, z_new = new_location.x.value, new_location.y.value, new_location.z.value
                if np.linalg.norm(np.array([x_new, y_new, z_new]) - np.array([x, y, z])) < 0.1:
                    # print(f'converged after {i + 1}')
                    break

                # Update for next iteration
                projected_ellipsoid = new_location
                x, y, z = x_new, y_new, z_new
                normal = np.array([x, y, z * (WGS84_SEMIMAJOR_AXIS / WGS84_SEMIMINOR_AXIS) ** 2])
                normal = normal / np.linalg.norm(normal)

            # Convert to EarthLocation (lat, lon, height)
            earth_location = earth_itrs.earth_location
            # print(np.linalg.norm(earth_location.get_itrs().cartesian.xyz - projected_ellipsoid.get_itrs().cartesian.xyz))

            if PLOT:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

                # print(moon_gcrs.cartesian.xyz.to(u.m).value)
                # print(*surface_coord.cartesian.xyz.to(u.m).value)
                # ax.plot(*surface_coord.cartesian.xyz.to(u.m).value)
                # plot_sphere(ax, radius=MOON_RADIUS.to(u.m).value, offset=moon_gcrs.cartesian.xyz.to(u.m).value)
                plot_sphere(ax, radius=earth_radius)
                distances = np.array([np.linspace(3e8, 5e8, 10)])
                projected_line = pos + (direction * distances.T)
                ax.plot(*projected_line.T)
                # ax.quiver([0], [0], [0], *(star_unit * 1e8))
                new_earth_intersection = pos + t1 * direction
                ax.plot(*new_earth_intersection, 'ko')
                new_earth_intersection = pos + t2 * direction
                ax.plot(*new_earth_intersection, 'ko')
                ax.plot(*earth_intersection, 'ro')

                ax.axes.set_xlim(-earth_radius, earth_radius)
                ax.axes.set_ylim(-earth_radius, earth_radius)
                ax.axes.set_zlim(-earth_radius, earth_radius)

                ax.set_aspect("equal")
                plt.show()

            ellipsoid_point_time = perf_counter_ns()

            total_azimuth_time = ellipsoid_point_time - azi_start
            ellipsoid_time = ellipsoid_point_time - sphere_point_time
            grazing_time = max_vec_time - original_vec_time

            shadow_locations.append(earth_location)
            timings.append({
                'total_time': total_azimuth_time,
                'sphere_time': sphere_point_time - max_vec_time,
                'grazing_time': grazing_time,
                'ellipsoid_time': ellipsoid_time,
            })

        else:
            # Without a DEM, assume a spherical moon
            elevation = 0 * u.km

        elevations[i] = elevation

    timings_df = pd.DataFrame(timings)
    print(timings_df.median())

    return azimuths, elevations, shadow_locations


def find_grazing_zone(star_coord, obs_time, moon_radius=1737.1 * u.km, dem_file=None):
    """
    Find the band of altitudes on the lunar surface that need to be considered
    for grazing occultation calculations.

    Parameters:
    -----------
    star_coord : SkyCoord
        Coordinates of the star in ICRS frame
    obs_time : Time
        Observation time
    moon_radius : Quantity
        Lunar reference radius
    dem_file : str, optional
        Path to lunar DEM TIFF file

    Returns:
    --------
    zone_center : MoonLocation
        Location on the Moon closest to the grazing zone center
    zone_width : Quantity
        Width of the grazing zone in km
    """
    # Convert star coordinates to MCMF frame
    star_mcmf = star_coord.transform_to(MCMF(obstime=obs_time))

    # Get the unit vector pointing from Moon center toward the star
    star_vec = np.array([star_mcmf.cartesian.x.value[0],
                         star_mcmf.cartesian.y.value[0],
                         star_mcmf.cartesian.z.value[0]])
    star_vec = star_vec / np.linalg.norm(star_vec)

    # The grazing zone center is perpendicular to the star direction
    # We need to find the point on the lunar surface where the star is at the horizon

    # Find a vector that's perpendicular to the star direction
    if np.abs(star_vec[2]) < 0.9:
        perp_vec = np.cross(star_vec, np.array([0, 0, 1]))
    else:
        perp_vec = np.cross(star_vec, np.array([1, 0, 0]))
    perp_vec = perp_vec / np.linalg.norm(perp_vec)

    # The center of the grazing zone is approximately in this direction
    grazing_center_vec = perp_vec * moon_radius.value

    # Convert to MoonLocation
    zone_center = MoonLocation.from_selenocentric(
        grazing_center_vec[0] * u.km,
        grazing_center_vec[1] * u.km,
        grazing_center_vec[2] * u.km
    )

    # Estimate the width of the zone (this is approximate and depends on many factors)
    # For grazing occultations, usually interested in terrain within a few km of the limb
    zone_width = 10 * u.km  # Adjust based on precision needs and lunar terrain

    return zone_center, zone_width


# Example usage
def main():
    star = SkyCoord.from_name('Antares')
    dem_file = "./Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

    timeseries = TimeSeries(time_start='2025-08-31T11:22:00Z', time_delta=2 * u.minute, n_samples=30)

    out_dict = {}

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=170))
    ax.coastlines()

    # Load DEM if provided
    if dem_file:
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)

    for row in timeseries:
        obs_time = row['time']
        zone_center, zone_width = find_grazing_zone(star, obs_time)

        print(f"Grazing zone center: {zone_center.lat}, {zone_center.lon}")
        print(f"Grazing zone width: {zone_width}")

        azimuths, elevations, shadow_points = calculate_lunar_horizon(star, obs_time, dem_data)

        lats = []
        lons = []

        for point in shadow_points:
            lon, lat, height = point.geodetic
            lons.append(lon.to(u.deg).value)
            lats.append(lat.to(u.deg).value)

        out_dict[obs_time.value] = np.array([lons, lats])

    plt.show()

    with open('edges.pickle', 'wb') as f:
        pickle.dump(out_dict, f)

    # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.polar(azimuths.to_value(u.rad), elevations.to_value(u.km) + 1737.1)
    # plt.title(f"Lunar Horizon Profile for Star Occultation on {obs_time.iso}")
    # plt.grid(True)
    # plt.savefig("lunar_horizon_profile.png")
    # plt.show()


if __name__ == "__main__":
    from pathlib import Path
    if Path("edges.pickle").is_file() and False:
        with open("edges.pickle", "rb") as f:
            edges = pickle.load(f)

        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=170.5, central_latitude=-43))
        ax.coastlines()
        for time, [lons, lats] in edges.items():
            ax.plot(lons, lats, c='k', alpha=0.3, transform=ccrs.Geodetic())

        # ax.set_global()

        plt.show()
    else:
        main()