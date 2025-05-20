from multiprocessing import shared_memory
import multiprocessing as mp
from functools import partial
from time import perf_counter_ns, sleep
import matplotlib.pyplot as plt
import pickle
from cartopy.crs import WGS84_SEMIMAJOR_AXIS, WGS84_SEMIMINOR_AXIS
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rasterio
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS, ITRS, CartesianRepresentation, EarthLocation
from lunarsky import SkyCoord, MoonLocation, Time, MCMF
from astropy.timeseries import TimeSeries
import cartopy.crs as ccrs
from tqdm import tqdm
import logging
import multiprocessing_logging


MOON_RADIUS=1737.4 * u.km
PLOT=False

# points_projected = mp.Value('i', 0)


def plot_sphere(ax, radius, offset=[0,0,0]):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe((x * radius) + offset[0], (y * radius) + offset[1], (z * radius) + offset[2], color="r")


def optimized_ellipsoid_intersection(earth_itrs, earth_gcrs, direction, obs_time, max_iterations=10, convergence_threshold=1):
    projected_ellipsoid = EarthLocation.from_geodetic(
        earth_itrs.earth_location.geodetic.lon,
        earth_itrs.earth_location.geodetic.lat,
        height=0 * u.m
    )

    # Do the initial transform once
    projected_gcrs = projected_ellipsoid.get_itrs(obstime=obs_time).transform_to(GCRS(obstime=obs_time))

    # Pre-calculate constants
    earth_position = earth_gcrs.cartesian.xyz.value
    ellipsoid_factor = (WGS84_SEMIMAJOR_AXIS / WGS84_SEMIMINOR_AXIS) ** 2

    # Current position
    position = projected_gcrs.cartesian.xyz.value

    # Iterative refinement with early stopping
    for i in range(max_iterations):
        # Calculate ellipsoid normal more efficiently
        normal = np.array([position[0], position[1], position[2] * ellipsoid_factor])
        normal = normal / np.linalg.norm(normal)

        # Ray-plane intersection
        numerator = np.dot(position - earth_position, normal)
        denominator = np.dot(direction, normal)
        t = numerator / denominator

        # New intersection point
        intersection = earth_position + t * direction

        # Transform to ITRS only once per iteration
        new_gcrs = GCRS(CartesianRepresentation(x=intersection[0], y=intersection[1], z=intersection[2], unit=u.m),
                        obstime=obs_time)
        new_itrs = new_gcrs.transform_to(ITRS(obstime=obs_time))
        new_location = new_itrs.earth_location

        # Project to surface
        new_surface = EarthLocation.from_geodetic(
            new_location.lon,
            new_location.lat,
            height=0 * u.m
        )

        # Transform back to GCRS
        new_gcrs = new_surface.get_itrs(obstime=obs_time).transform_to(GCRS(obstime=obs_time))
        new_position = new_gcrs.cartesian.xyz.value

        # Check convergence
        if np.linalg.norm(new_position - position) < convergence_threshold:
            return new_gcrs

        # Update for next iteration
        position = new_position

    # Return the final position if we didn't converge
    return GCRS(CartesianRepresentation(x=position[0], y=position[1], z=position[2], unit=u.m), obstime=obs_time)

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

spice_lock = mp.Lock()

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
    with spice_lock:
        star_mcmf = star_coord.transform_to(MCMF(obstime=obs_time))

    # Define azimuths around the lunar limb
    azimuths = np.linspace(0, 360, resolution, endpoint=False) * u.deg

    # Initialize array to store maximum elevations
    elevations = np.zeros(len(azimuths)) * u.km

    shadow_locations = []
    timings = []

    # For each azimuth, find the point on the lunar limb
    for i, az in enumerate(azimuths):
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

            distances = (np.arange(start=-400, stop=400, step=0.1) * u.km).to(u.m).value.reshape(-1, 1)
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


            # with spice_lock:
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
            if (165 * u.degree < earth_itrs.earth_location.geodetic.lon < 175 * u.degree
                    and -48 * u.degree < earth_itrs.earth_location.geodetic.lat < -40 * u.degree):
                ellipsoid_point = optimized_ellipsoid_intersection(earth_itrs, earth_gcrs, direction, obs_time)
            else:
                ellipsoid_point = earth_gcrs

            ellipsoid_point_time = perf_counter_ns()

            # Convert to EarthLocation (lat, lon, height)
            earth_location = ellipsoid_point.transform_to(ITRS(obstime=obs_time)).earth_location

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

            shadow_locations.append(earth_location)

            # total_azimuth_time = ellipsoid_point_time - azi_start
            # ellipsoid_time = ellipsoid_point_time - sphere_point_time
            # grazing_time = max_vec_time - original_vec_time
            #
            # timings.append({
            #     'total_time': total_azimuth_time,
            #     'sphere_time': sphere_point_time - max_vec_time,
            #     'grazing_time': grazing_time,
            #     'ellipsoid_time': ellipsoid_time,
            # })

        else:
            # Without a DEM, assume a spherical moon
            elevation = 0 * u.km

        elevations[i] = elevation

        # with points_projected.get_lock():
        #     points_projected.value += 1

    # timings_df = pd.DataFrame(timings)
    # logging.debug(timings_df.mean() / 1e6)

    return azimuths, elevations, shadow_locations


def process_time_point(obs_time, star, shm_name, shape, dtype):
    """Process a time point using DEM data from shared memory."""
    # Attach to existing shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)

    # Create a NumPy array that references the shared memory
    dem_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Process using the shared data
    azimuths, elevations, shadow_points = calculate_lunar_horizon(star, obs_time, dem_data, resolution=360 * 60)

    lats = []
    lons = []

    for point in shadow_points:
        lon, lat, height = point.geodetic
        lons.append(lon.to(u.deg).value)
        lats.append(lat.to(u.deg).value)

    # Clean up the shared memory attachment (but don't unlink it)
    existing_shm.close()

    return obs_time.value, np.array([lons, lats])

def initialize_shared_memory(dem_file):
    """Load DEM data and create a shared memory block containing it."""
    # Load DEM data
    with rasterio.open(dem_file) as src:
        dem_data = src.read(1)

    # Get shape and data type information
    shape = dem_data.shape
    dtype = dem_data.dtype

    # Create a shared memory block large enough to contain the array
    shm = shared_memory.SharedMemory(create=True, size=dem_data.nbytes)

    # Create a NumPy array backed by shared memory
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Copy data into shared array
    shared_array[:] = dem_data[:]

    return shm.name, shape, dtype

# Example usage
def main():
    dem_file = "./Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

    timeseries = TimeSeries(time_start='2025-08-31T11:20:00Z', time_delta=10 * u.second, n_samples=6 * 60)

    # Create shared memory for DEM data
    print(f"Reading {dem_file}")
    shm_name, shape, dtype = initialize_shared_memory(dem_file)

    # Get the time points
    time_points = [row['time'] for row in timeseries]

    # Get the number of available CPU cores
    num_cores = 16
    num_workers = max(1, num_cores - 1)
    print(f"Starting {num_workers} workers")

    out_dict = {}

    logging.basicConfig(level=logging.DEBUG)
    multiprocessing_logging.install_mp_handler()

    try:
        for star_name in ['Antares', 'Antares B']:
            star = SkyCoord.from_name(star_name)
            with tqdm(total=len(time_points), desc=f"Processing for {star_name}") as pbar:
                # Create a pool of worker processes
                with mp.Pool(processes=num_workers) as pool:
                    # Create a partial function with the star and shared memory details
                    process_func = partial(process_time_point,
                                           star=star,
                                           shm_name=shm_name,
                                           shape=shape,
                                           dtype=dtype)

                    # Map the function to all time points
                    results = pool.imap_unordered(process_func, time_points)

                    #     prev_points = 0
                    #     while prev_points < pbar.total:
                    #         with points_projected.get_lock():
                    #             change = points_projected.value - prev_points
                    #             prev_points = points_projected.value
                    #         pbar.update(change)
                    #         sleep(0.1)

                    # Process results
                    for time_value, result in results:
                        out_dict[time_value] = result
                        pbar.update(1)

            with open(f'{star_name}_edges.pickle', 'wb') as f:
                pickle.dump(out_dict, f)
    finally:
        # Clean up the shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()  # This actually removes the shared memory block

    # Save results
    # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.polar(azimuths.to_value(u.rad), elevations.to_value(u.km) + 1737.1)
    # plt.title(f"Lunar Horizon Profile for Star Occultation on {obs_time.iso}")
    # plt.grid(True)
    # plt.savefig("lunar_horizon_profile.png")
    # plt.show()


if __name__ == "__main__":
    main()