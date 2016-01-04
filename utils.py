import numpy as np

def geo_rotate(point, new_pole):
    """ Rotates spherical coordinates to a frame where north pole
    is at the new_pole position """
    lat = point[0]
    lon = point[1]

    # Convert to radians
    lon = lon * np.pi / 180
    lat = lat * np.pi / 180

    # Get new pole coordinates
    south_lat = new_pole[0]
    south_lon = new_pole[1]

    theta = -90 + south_lat
    phi = south_lon

    theta = theta * np.pi / 180
    phi = phi * np.pi / 180

    # Convert to cartesian
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    # Rotate 
    x_new = np.cos(theta) * np.cos(phi) * x +\
            np.cos(theta) * np.sin(phi) * y +\
            np.sin(theta) * z

    y_new = -np.sin(phi) * x + np.cos(phi) * y

    z_new = -np.sin(theta) * np.cos(phi) * x -\
            np.sin(theta) * np.sin(phi) * y +\
            np.cos(theta) * z

    # Go back to spherical
    lon_new = np.arctan2(y_new, x_new)
    lat_new = np.arcsin(z_new)

    # Go back to geo
    lon_new = 180.0 * lon_new / np.pi
    lat_new = 180.0 * lat_new / np.pi

    return [lat_new, lon_new]
