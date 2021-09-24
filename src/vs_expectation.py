import numpy as np
import pickle
import json
from pathlib import Path
from config import *
from utils import load_visceral_slides, get_inspexp_frames
from stat import contour_mean_len, get_vs_range
import SimpleITK as sitk
from vs_definitions import VSTransform
from vis_visceral_slide import plot_visceral_slide_expectation
from cinemri.definitions import CineMRISlice


def get_regions_stat(visceral_slides,
                     vs_range,
                     regions_num=120,
                     transform=VSTransform.none,
                     plot=False,
                     inspexp_data=None,
                     images_path=None,
                     output_path=None):
    """
    Calculates means and standard deviations of visceral slide by relative position at the contour
    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       Visceral slides to calculate the statistics for
    vs_range : tuple
       (vs_min, vs_max) a range of visceral slide values to consider
    regions_num : int
       A number of regions to split visceral slide contour into
    transform : VSTransform, default=VSTransform.none
       A transformation to apply to visceral slide values
    plot : bool
       A boolean flag indicating whether to visualise the expectation
    inspexp_data : dict
       A dictionary containing infromation about inspiration/expiration frames positions
    images_path : Path
       A path to cine-MRI scans that correspond to visceral slides
    output_path : Path
       A path where to save visaulisation

    Returns
    -------
    stat : dict
       A dictionary that contains means and standard deviation, keys "means" and "stds"
    """
    vs_regions = np.array([vs.to_regions() for vs in visceral_slides])

    # Now calculate the statistics for each region
    means = []
    stds = []
    for i in range(regions_num):
        cur_regions = vs_regions[:, i]
        values = cur_regions[0].values
        for j in range(1, len(visceral_slides)):
            values = np.concatenate((values, cur_regions[j].values))

        # Remove the outliers
        values = np.array([value for value in values if vs_range[0] <= value <= vs_range[1]])

        # Apply transformation is specified
        if transform == VSTransform.log:
            values = np.log(values)
        elif transform == VSTransform.sqrt:
            values = np.sqrt(values)

        means.append(np.mean(values))
        stds.append(np.std(values))

    if plot:
        vs = visceral_slides[0]
        slice = CineMRISlice.from_full_id(vs.full_id)

        title = "cum_expectation_vi" if inspexp_data is None else "inspexp_expectation_vi"
        if transform == VSTransform.log:
            title += "_log"
        elif transform == VSTransform.sqrt:
            title += "_sqrt"
        title += "_sqrt" + str(regions_num)

        if inspexp_data is None:
            image = sitk.ReadImage(str(slice.build_path(images_path)))
            frame = sitk.GetArrayFromImage(image)[-2]
        else:
            frame, _ = get_inspexp_frames(slice, inspexp_data, images_path)

        plot_visceral_slide_expectation(means, vs, title, frame, output_path)

    return {"means": means, "stds": stds}


def calculate_and_save_vs_stat(vs_path,
                               masks_path,
                               images_path,
                               output_path,
                               inspexp_data=None,
                               transform=VSTransform.none,
                               point_in_chunk=5):
    """
    Calculates and saves visceral slide statistics
    Parameters
    ----------
    vs_path : Path
       A path to visceral slides to calculate the statistics for
    masks_path : Path
       A path to segmentation masks to calculate the average contour lenght
    images_path : Path
       A path to cine-MRI scans that correspond to visceral slides
    output_path : Path
       A path where to save a pickle file with statistics and plots
    inspexp_data : dict
       A dictionary containing information about inspiration/expiration frames positions
    transform : VSTransform, default=VSTransform.none
       A transformation to apply to visceral slide values
    point_in_chunk : int
       A target number of points per region
    """

    output_path = output_path / "vs_expectation"
    output_path.mkdir(exist_ok=True)

    visceral_slides = load_visceral_slides(vs_path)
    vs_min, vs_max = get_vs_range(visceral_slides, False)
    avg_contour_len = contour_mean_len(masks_path)
    chunks_num = round(avg_contour_len / point_in_chunk)

    expectation = get_regions_stat(visceral_slides,
                                   (vs_min, vs_max),
                                   chunks_num,
                                   transform,
                                   True,
                                   inspexp_data,
                                   images_path,
                                   output_path)

    pkl_title = "cumulative_vs_expectation" if inspexp_data is None else "inspexp_vs_expectation"
    if transform == VSTransform.log:
        pkl_title += "_log"
    elif transform == VSTransform.sqrt:
        pkl_title += "_sqrt"

    with open(pkl_title + ".pkl", "w+b") as f:
        pickle.dump(expectation, f)

