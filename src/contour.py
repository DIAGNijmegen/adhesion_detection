import numpy as np
from cinemri.contour import AbdominalContourPart, Contour


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

    def get_distance(point1, point2, axis=-1):
        if axis == -1:
            distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        else:
            distance = abs(point1[axis] - point2[axis])

        return distance

    for index in range(1, coords_num):
        coord_curr = contour_subset_coords[index]
        distance = get_distance(coord_curr, coord_prev, axis=axis)

        if distance > connectivity_threshold:
            regions.append(np.array(current_region))
            current_region = []
        coord_prev = coord_curr
        current_region.append(coord_prev)

    # Check if first and last region should be merged
    if len(regions) > 0:
        if len(current_region) > 0:
            first_of_first = regions[0][0]
            last_of_last = current_region[-1]

            distance = get_distance(first_of_first, last_of_last, axis=axis)

            if distance > connectivity_threshold:
                # First and last regions are separated
                regions.append(np.array(current_region))
            else:
                # First and last regions are connected
                full_region = np.concatenate((current_region, regions[0]))
                regions[0] = full_region
    else:
        regions.append(np.array(current_region))

    return regions


def get_adhesions_prior_coords(x, y):
    prior_coords = np.column_stack((x, y))

    contour = Contour(x, y)

    # Remove top coordinates
    x_top, y_top = contour.get_abdominal_contour_part(AbdominalContourPart.top)
    top_coords = np.column_stack((x_top, y_top)).tolist()
    prior_coords = [coord for coord in prior_coords.tolist() if coord not in top_coords]

    # remove pelvis
   # x_bottom, y_bottom = contour.get_abdominal_contour_part(AbdominalContourPart.bottom)
   # pelvis_coords = np.column_stack((x_bottom, y_bottom)).tolist()
   # prior_coords = [coord for coord in prior_coords if coord not in pelvis_coords]

    # We remove top 1/2 of anterior wall coordinates
    x_anterior_wall, y_anterior_wall = contour.get_abdominal_contour_part(AbdominalContourPart.anterior_wall)
    anterior_wall_coords = np.column_stack((x_anterior_wall, y_anterior_wall))
    y_anterior_wall_cutoff = sorted(y_anterior_wall)[int(len(y_anterior_wall) / 2)]
    anterior_wall_coords = [coord for coord in anterior_wall_coords.tolist() if coord[1] < y_anterior_wall_cutoff]
    prior_coords = [coord for coord in prior_coords if coord not in anterior_wall_coords]

    # We remove posterior wall coordinates
    x_posterior_wall, y_posterior_wall = contour.get_abdominal_contour_part(AbdominalContourPart.posterior_wall)
    posterior_wall_coords = np.column_stack((x_posterior_wall, y_posterior_wall))
    prior_coords = np.array([coord for coord in prior_coords if coord not in posterior_wall_coords.tolist()])

    return prior_coords[:, 0], prior_coords[:, 1]

from pathlib import Path
from cinemri.config import ARCHIVE_PATH
from utils import load_visceral_slides
import SimpleITK as sitk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    archive_path = ARCHIVE_PATH
    images_path = archive_path / "detection_new" / "images"
    cumulative_vs_path = Path("../../data/vs_cum/cumulative_vs_contour_reg_det_full_df")

    visceral_slides = load_visceral_slides(cumulative_vs_path)
    output_path = Path("anchor_points_contour_parts")
    output_path.mkdir(exist_ok=True)
    for i in range(len(visceral_slides)):
        vs = visceral_slides[i]
        vs_image = vs.build_path(images_path)
        image = sitk.ReadImage(str(vs_image))
        frame = sitk.GetArrayFromImage(image)[-2]

        contour = Contour(vs.x, vs.y)
        x1, y1 = contour.get_abdominal_contour_part(AbdominalContourPart.anterior_wall)
        x2, y2 = contour.get_abdominal_contour_part(AbdominalContourPart.top)
        x3, y3 = contour.get_abdominal_contour_part(AbdominalContourPart.posterior_wall)
        x4, y4 = contour.get_abdominal_contour_part(AbdominalContourPart.bottom)

        plt.figure()
        plt.imshow(frame, cmap="gray")
        plt.scatter(vs.x, vs.y, s=5, color="b")
        plt.scatter(x1, y1, s=10, color="r")
        plt.scatter(x2, y2, s=10, color="y")
        plt.scatter(x3, y3, s=10, color="g")
        plt.scatter(x4, y4, s=10, color="c")

        plt.savefig(output_path / "{}.png".format(vs.full_id), bbox_inches='tight', pad_inches=0)
        plt.close()


