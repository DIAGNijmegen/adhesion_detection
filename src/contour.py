import numpy as np
from cinemri.contour import AbdominalContourPart

def get_connected_regions(contour_subset_coords, connectivity_threshold=5, axis=-1):
    """
    Given a subset of contour coordinates returns a list of connected regions
    considering the specified connectivity threshold

    Parameters
    ----------
    contour_subset_coords : list of list
       A list containing a subset of coordinates of a contour
    connectivity_threshold : int, default=5
       Threshold which indicates the maximum difference in x component allowed between
       the subsequent coordinates of a contour to be considered connected

    Returns
    -------
    regions : list of list
       A list of lists of coordinates that belong to connected regions. Length of regions might vary

    """

    regions = []
    coord_prev = contour_subset_coords[0]
    coords_num = contour_subset_coords.shape[0]
    current_region = [coord_prev]
    for index in range(1, coords_num):
        coord_curr = contour_subset_coords[index]
        if axis == -1:
            distance = np.sqrt((coord_curr[0] - coord_prev[0]) ** 2 + (coord_curr[1] - coord_prev[1]) ** 2)
        else:
            distance = abs(coord_curr[axis] - coord_prev[axis])

        if distance > connectivity_threshold:
            regions.append(current_region)
            current_region = []
        coord_prev = coord_curr
        current_region.append(coord_prev)

    regions.append(current_region)
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
        # Get all x values of coordinates which have x = x_current
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
        if region[0][1] < middle_y and region[-1][1] < middle_y:
            top_regions.append(region)

    # Find and return the longest region
    top_regions_len = np.array([(region[-1][0] - region[0][0] + 1) for region in top_regions])
    longest_region_ind = np.argmax(top_regions_len)
    longest_region = top_regions[longest_region_ind]

    longest_region_start = longest_region[0]
    longest_region_end = longest_region[-1]

    # The contour returned by cinemri.contour.get_contour() always starts from
    # the lowest-left point of the contour, so when we determine the top part
    # we can safely expect that by using the start and end point
    # of the longest top region we will get a connected area from the contour coordinates

    longest_region_start_ind = np.where((coords == longest_region_start).all(axis=1))[0][0]
    longest_region_end_ind = np.where((coords == longest_region_end).all(axis=1))[0][0]
    top_coords = coords[longest_region_start_ind:longest_region_end_ind]

    return top_coords[:, 0], top_coords[:, 1]


def get_abdominal_wall_coord(x, y, type=AbdominalContourPart.anterior_wall, connectivity_threshold=5):
    """
    Extracts coordinates of the abdominal wall of the specified type from abdominal cavity contour

    Parameters
    ----------
    x, y : list of int
       x-axis and y-axis components of abdominal cavity contour
    type : AbdominalContourPart
       indicates which abdominal wall to detect, anterior or posterior

    connectivity_threshold: int, default=5
       Threshold which indicates the maximum difference in x component allowed between
       the subsequent coordinates of a contour to be considered connected

    Returns
    -------
    x_wall, y_wall : ndarray of int
       x-axis and y-axis components of abdominal wall
    """

    expected_types = [AbdominalContourPart.anterior_wall, AbdominalContourPart.posterior_wall]
    if type not in expected_types:
        raise ValueError("Either AbdominalContourPart.anterior_wall or AbdominalContourPart.posterior_wall "
                         "should be provided as abdominal wall type")

    coords = np.column_stack((x, y))
    # Get unique y coordinates
    y_unique = np.unique(y)
    # For each unique y find the x on the specified side of abdominal cavity contour
    x_side = []
    desc_sort = type == AbdominalContourPart.posterior_wall
    for y_current in y_unique:
        # Get all x values of coordinates which have y = y_current
        xs = [coord[0] for coord in coords if coord[1] == y_current]
        x_side.append(sorted(xs, reverse=desc_sort)[0])

    # Unique y coordinates and the corresponding x coordinates
    # should give good approximation of the abdominal wall
    x_side = np.array(x_side)
    abdominal_wall_coords = np.column_stack((x_side, y_unique))
    #return abdominal_wall_coords[:, 0], abdominal_wall_coords[:, 1]

    regions = get_connected_regions(abdominal_wall_coords, connectivity_threshold, axis=0)

    # Find and return the longest region
    top_regions_len = np.array([(region[-1][1] - region[0][1] + 1) for region in regions])
    longest_region_ind = np.argmax(top_regions_len)
    longest_region = regions[longest_region_ind]

    # TODO: make this code more robust
    # Handling of different orders of coordinates list
    if type == AbdominalContourPart.anterior_wall:
        longest_region_start = longest_region[-1]
        longest_region_end = longest_region[0]
    else:
        longest_region_start = longest_region[0]
        longest_region_end = longest_region[-1]

    longest_region_start_ind = np.where((coords == longest_region_start).all(axis=1))[0][0]
    longest_region_end_ind = np.where((coords == longest_region_end).all(axis=1))[0][0]

    # If the end index of the region is 0 in case of the posterior wall
    # it means that the first point of the contour was identified as the last one
    # hence we should take the last contour point to get correct part of the contour
    if type == AbdominalContourPart.posterior_wall and longest_region_end_ind == 0:
        longest_region_end_ind = len(coords) - 1

    abdominal_wall_coords = coords[longest_region_start_ind:longest_region_end_ind]
    return abdominal_wall_coords[:, 0], abdominal_wall_coords[:, 1]


def get_adhesions_prior_coords(x, y, connectivity_threshold=5):
    coords = np.column_stack((x, y))

    # Keep only the top half of the contour

    middle_y = (np.max(y) - np.min(y)) / 2
    coords_top = np.array([coord for coord in coords if coord[1] <= middle_y])

    # Get the top part of the anterior wall
    anterior_wall_x, anterior_wall_y = get_abdominal_wall_coord(coords_top[:, 0],
                                                                coords_top[:, 1],
                                                                connectivity_threshold=connectivity_threshold)
    anterior_wall_coords = np.column_stack((anterior_wall_x, anterior_wall_y)).tolist()

    # Also discard the top quarter of the anterior wall
    middle_anterior_y = (np.max(anterior_wall_y) - np.min(anterior_wall_y)) / 2
    anterior_wall_bottom = [coord for coord in anterior_wall_coords if coord[1] > middle_anterior_y]

    # Get the top right part of the contour
    rest_coords = [coord for coord in coords_top.tolist() if coord not in anterior_wall_bottom]
    # Filter out the top right part of the contour
    adhesions_prior_coords = [coord for coord in coords.tolist() if coord not in rest_coords]

    # Get the top coordinates
    top_x, top_y = get_abdominal_contour_top(coords[:, 0],
                                             coords[:, 1],
                                             connectivity_threshold=connectivity_threshold)
    top_coords = np.column_stack((top_x, top_y)).tolist()
    # Filter out the top part of the contour
    adhesions_prior_coords = np.array([coord for coord in adhesions_prior_coords if coord not in top_coords])

    return adhesions_prior_coords[:, 0], adhesions_prior_coords[:, 1]


