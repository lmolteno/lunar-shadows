import numpy as np

def generate_spherical_gridlines_to_3d_array(
        radius=1.0, num_lat_lines=5, num_lon_lines=10, resolution=50
):
    """
    Generates gridlines on a sphere and returns them as a single 3D NumPy array.
    The array shape will be (num_lat_lines + num_lon_lines, resolution, 3).
    Latitude lines are stacked first, then longitude lines.

    Args:
        radius (float): The radius of the sphere.
        num_lat_lines (int): The number of latitude lines (parallels, excluding poles).
        num_lon_lines (int): The number of longitude lines (meridians).
        resolution (int): The number of points to define each individual line.

    Returns:
        numpy.ndarray: A 3D array of shape (total_lines, resolution, 3)
                       containing the (x, y, z) coordinates of the gridlines.
                       Handles cases where num_lines or resolution are zero
                       by returning an appropriately shaped empty array.
    """
    # Handle zero resolution case first
    if resolution <= 0:  # If resolution is 0 or negative, points cannot be defined
        return np.empty((num_lat_lines + num_lon_lines, 0, 3))

    # Handle case where no lines are requested
    if num_lat_lines <= 0 and num_lon_lines <= 0:
        return np.empty((0, resolution, 3))

    all_lines_list = []

    # --- Latitude Lines (Parallels) ---
    # For these lines, phi (polar angle) is constant for each line, theta (azimuth) varies
    if num_lat_lines > 0:
        # Phi values for each latitude line (polar angle from Z-axis, 0 at North Pole)
        # Exclude poles (0 and pi) for continuous lines by slicing [1:-1]
        phi_for_lat_lines = np.linspace(0, np.pi, num_lat_lines + 2)[1:-1]  # Shape: (num_lat_lines,)

        # Theta values for points along each latitude line (azimuthal angle from X-axis)
        theta_along_line = np.linspace(0, 2 * np.pi, resolution)  # Shape: (resolution,)

        # Expand dimensions for broadcasting:
        # phi_for_lat_lines becomes (num_lat_lines, 1)
        # theta_along_line remains (resolution,)
        sin_phi = np.sin(phi_for_lat_lines[:, np.newaxis])  # Shape: (num_lat_lines, 1)
        cos_phi = np.cos(phi_for_lat_lines[:, np.newaxis])  # Shape: (num_lat_lines, 1)

        sin_theta = np.sin(theta_along_line)  # Shape: (resolution,)
        cos_theta = np.cos(theta_along_line)  # Shape: (resolution,)

        # Calculate Cartesian coordinates (broadcasting applies)
        # (num_lat_lines, 1) * (resolution,) -> (num_lat_lines, resolution)
        x_lat = radius * sin_phi * cos_theta
        y_lat = radius * sin_phi * sin_theta
        # z is constant for a given latitude line, determined by cos_phi
        # cos_phi is (num_lat_lines, 1), needs to be (num_lat_lines, resolution)
        z_lat = radius * cos_phi * np.ones_like(theta_along_line[np.newaxis, :])  # Or np.ones((1, resolution))

        # Stack coordinates to form points for latitude lines
        lat_lines_points = np.stack((x_lat, y_lat, z_lat), axis=-1)  # Shape: (num_lat_lines, resolution, 3)
        all_lines_list.append(lat_lines_points)

    # --- Longitude Lines (Meridians) ---
    # For these lines, theta (azimuth) is constant for each line, phi (polar angle) varies
    if num_lon_lines > 0:
        # Theta values for each longitude line
        theta_for_lon_lines = np.linspace(0, 2 * np.pi, num_lon_lines, endpoint=False)  # Shape: (num_lon_lines,)

        # Phi values for points along each longitude line (from North Pole to South Pole)
        phi_along_line = np.linspace(0, np.pi, resolution)  # Shape: (resolution,)

        # Expand dimensions for broadcasting:
        # theta_for_lon_lines becomes (num_lon_lines, 1)
        # phi_along_line remains (resolution,)
        sin_theta = np.sin(theta_for_lon_lines[:, np.newaxis])  # Shape: (num_lon_lines, 1)
        cos_theta = np.cos(theta_for_lon_lines[:, np.newaxis])  # Shape: (num_lon_lines, 1)

        sin_phi = np.sin(phi_along_line)  # Shape: (resolution,)
        cos_phi_val = np.cos(phi_along_line)  # Shape: (resolution,)

        # Calculate Cartesian coordinates (broadcasting applies)
        # (resolution,) * (num_lon_lines, 1) -> (num_lon_lines, resolution)
        x_lon = radius * sin_phi * cos_theta
        y_lon = radius * sin_phi * sin_theta
        # z is determined by cos_phi_val (shape: (resolution,))
        # Needs to be tiled to match (num_lon_lines, resolution)
        z_lon_base = radius * cos_phi_val
        z_lon = np.tile(z_lon_base, (num_lon_lines, 1))  # Shape: (num_lon_lines, resolution)

        # Stack coordinates to form points for longitude lines
        lon_lines_points = np.stack((x_lon, y_lon, z_lon), axis=-1)  # Shape: (num_lon_lines, resolution, 3)
        all_lines_list.append(lon_lines_points)

    # Concatenate all generated line arrays (if any)
    if not all_lines_list:
        # This case should ideally be covered by the initial checks,
        # but as a fallback if, for example, num_lat_lines > 0 but becomes 0 after slicing.
        # However, given num_lat_lines > 0 check, linspace(0,pi,num_lat_lines+2)[1:-1] will be non-empty.
        # This path is mainly for logical completeness if initial checks were different.
        # The primary scenario for an empty list here (given resolution > 0) is
        # if num_lat_lines and num_lon_lines were passed as <= 0.
        # That is already handled by the second `if` block at the start.
        # So, all_lines_list should contain at least one array if we reach here.
        # For safety, if it somehow is empty:
        return np.empty((0, resolution, 3))

    return np.concatenate(all_lines_list, axis=0)