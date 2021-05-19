import numpy as np

# TODO: pass axis index
def get_connected_regions(contour_subset_coords, connectivity_threshold=5, axis=0):
    """
    Given a subset of contour coordinates returns a list of connected regions
    considering the specified connectivity threshold

    Parameters
    ----------
    contour_subset_coords : list of list
       A list containing a subset of coordinates of a contour
    connectivity_threshold : int, default=5
       Threshold which indicates the maximum difference in the specified axis component allowed between
       the subsequent coordinates of a contour to be considered connected
    axis : int, default=0
       An axis to verify connectivity. 0 - x, 1 - y

    Returns
    -------
    regions : list of list
       A list of start and end coordinates of all connected regions

    """

    regions = []
    region_start = contour_subset_coords[0]
    coord_prev = region_start
    coords_num = contour_subset_coords.shape[0]
    for index in range(1, coords_num):
        coord_curr = contour_subset_coords[index]
        if abs(coord_curr[axis] - coord_prev[axis]) > connectivity_threshold:
            regions.append([region_start, coord_prev])
            region_start = coord_curr
        coord_prev = coord_curr

    regions.append([region_start, contour_subset_coords[-1]])

    return regions


def get_abdominal_contour_top(x, y, connectivity_threshold=5):
    """
    Extracts coordinates of the abdominal wall of the specified type from abdominal cavity contour

    Parameters
    ----------
    x, y : list of int
       x-axis and y-axis components of abdominal cavity contour
    type : AbdominalContourPart
       indicates which abdominal wall to detect, anterior or posterior

    connectivity_threshold : int, default=5
       Threshold which indicates the maximum difference in x component allowed between
       the subsequent coordinates of a contour to be considered connected

    Returns
    -------
    x_wall, y_wall : ndarray of int
       x-axis and y-axis components of abdominal wall
    """

    coords = np.column_stack((x, y))
    # Get unique x coordinates
    x_unique = np.unique(x)

    # For each unique y find the x on the specified side of abdominal cavity contour
    y_top = []
    for x_current in x_unique:
        # Get all x values of coordinates which have y = y_current
        ys = [coord[1] for coord in coords if coord[0] == x_current]
        y_top.append(sorted(ys)[0])

    # Unique y coordinates and the corresponding x coordinates
    # should give good approximation of the abdominal wall
    y_top = np.array(y_top)
    top_contour_coords = np.column_stack((x_unique, y_top))

    regions = get_connected_regions(top_contour_coords, connectivity_threshold, axis=1)

    # Discard regions below the middle y
    y = np.array(y)
    middle_y = (y.max() - y.min()) / 2
    top_regions = []
    for region in regions:
        # Check that min and max y of a region is above middle_y
        if region[0][1] < middle_y and region[1][1] < middle_y:
            top_regions.append(region)

    # Find and return the longest region
    top_regions_len = np.array([(region[1][0] - region[0][0] + 1) for region in top_regions])
    longest_region_ind = np.argmax(top_regions_len)
    longest_region = top_regions[longest_region_ind]

    longest_region_start = longest_region[0]
    longest_region_end = longest_region[1]

    # The contour returned by cinemri.contour.get_contour() always starts from
    # the lowest-left point of the contour, so when we determine the top part
    # we can safely expect that by using the start and end point
    # of the longest top region we will get a connected area from the contour coordinates

    longest_region_start_ind = np.where((coords == longest_region_start).all(axis=1))[0][0]
    longest_region_end_ind = np.where((coords == longest_region_end).all(axis=1))[0][0]
    top_coords = coords[longest_region_start_ind:longest_region_end_ind]

    return top_coords[:, 0], top_coords[:, 1]