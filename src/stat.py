# Functions to calculate the statistics of the dataset
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils import slice_complete_and_sagittal
from cinemri.utils import get_patients
from cinemri.contour import get_contour


def find_unique_shapes(images_path):
    """Finds unique shapes of slices in the archive

    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images

    Returns
    -------
    shapes : list
    A list of unique shapes of slices in the archive
    """
    shapes = []

    patients = get_patients(images_path)
    for patient in patients:

        for cinemri_slice in patient.cinemri_slices:
            slice_image_path = cinemri_slice.build_path(images_path)
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_image_path)))[0]

            if not (image.shape in shapes):
                shapes.append(image.shape)

    return shapes


def get_vs_range(visceral_slides, negative_vs_needed, verbose=True):
    """Returns visceral slide range with excluded outliers to consider for adhesion prediction

    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       A list of visceral slides to calculate a range for
    negative_vs_needed : bool
       Boolean flag indicating that only negative visceral slide values are considered
       True for normalisation with standardisation, False for other normalisation options

    Returns
    -------
    min, max : float
    A range of visceral slide to consider for adhesion prediction
    """

    # Statistics useful for prediction
    all_vs_values = []
    for visceral_slide in visceral_slides:
        all_vs_values.extend(visceral_slide.values)

    vs_abs_min = np.min(all_vs_values)
    vs_abs_max = np.max(all_vs_values)
    if verbose:
        print("VS minumum : {}".format(vs_abs_min))
        print("VS maximum : {}".format(vs_abs_max))

    vs_q1 = np.quantile(all_vs_values, 0.25)
    vs_q3 = np.quantile(all_vs_values, 0.75)
    vs_iqr = vs_q3 - vs_q1

    if verbose:
        print("VS first quantile : {}".format(vs_q1))
        print("VS third quantile : {}".format(vs_q3))
        print("VS IQR : {}".format(vs_iqr))

    vs_min = min(vs_abs_min, vs_q1 - 1.5 * vs_iqr) if negative_vs_needed else max(vs_abs_min, vs_q1 - 1.5 * vs_iqr)
    vs_max = min(vs_abs_max, vs_q3 + 1.5 * vs_iqr)

    if verbose:
        print("VS minumum, outliers removed range : {}".format(vs_min))
        print("VS maximum, outliers removed range : {}".format(vs_max))

    return (vs_min, 0) if negative_vs_needed else (vs_min, vs_max)


def get_avg_contour_size(visceral_slides):
    """
    The average contour size for the set of visceral slides
    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       A list of visceral slide to compute the average

    Returns
    -------
    width, height : int
       Average contour size
    """
    widths = []
    heights = []

    for vs in visceral_slides:
        widths.append(vs.width)
        heights.append(vs.height)

    return round(np.mean(widths)), round(np.mean(heights))


def bb_size_stat(annotations):
    """
    Computes an average adhesion bounding box size and its standard deviation

    Parameters
    ----------
    annotations : list of AdhesionAnnotation
       List of annotations for which an average size of a bounding box should be determined

    Returns
    -------
    average_size, size_std : tuple of float
       An average adhesion bounding box size and its standard deviation
    """

    widths = []
    heights = []

    for annotation in annotations:
        for adhesion in annotation.adhesions:
            widths.append(adhesion.width)
            heights.append(adhesion.height)

    average_size = np.mean(widths), np.mean(heights)
    size_std = np.std(widths), np.std(heights)
    return average_size, size_std


def adhesions_stat(annotations):
    """
    Reports statistics of bounding box annotations

    Parameters
    ----------
    annotations : list of AdhesionAnnotation
        List of annotations for which statistics should be computed
    """

    widths = []
    heights = []
    largest_bb = annotations[0].adhesions[0]
    smallest_bb = annotations[0].adhesions[0]

    for annotation in annotations:
        for adhesion in annotation.adhesions:
            widths.append(adhesion.width)
            heights.append(adhesion.height)

            curr_perim = adhesion.width + adhesion.height

            if curr_perim > largest_bb.width + largest_bb.height:
                largest_bb = adhesion

            if curr_perim < smallest_bb.width + smallest_bb.height:
                smallest_bb = adhesion

    print("Minimum width: {}".format(np.min(widths)))
    print("Minimum height: {}".format(np.min(heights)))

    print("Maximum width: {}".format(np.max(widths)))
    print("Maximum height: {}".format(np.max(heights)))

    print("Mean width: {}".format(np.mean(widths)))
    print("Mean height: {}".format(np.mean(heights)))

    print("Median width: {}".format(np.median(widths)))
    print("Median height: {}".format(np.median(heights)))

    print("Width std: {}".format(np.std(widths)))
    print("Height std: {}".format(np.std(heights)))

    plt.figure()
    plt.boxplot(widths)
    plt.title("Widths")
    plt.show()

    plt.figure()
    plt.boxplot(heights)
    plt.title("Heights")
    plt.show()

    print("Smallest annotation, x_o: {}, y_o: {}, width: {}, height: {}".format(smallest_bb.origin_x,
                                                                                smallest_bb.origin_y, smallest_bb.width,
                                                                                smallest_bb.height))
    print("Largest annotation, width: {}, height: {}".format(largest_bb.width, largest_bb.height))


def contour_mean_len(masks_path):
    """
    Calculates mean length of abdominal cavity contour
    Parameters
    ----------
    masks_path : Path
       A path to abdominal cavity segmentations of cine-MRI slices to calculate mean contour length for

    Returns
    -------
    mean_length : float
       The mean abdominal cavity contour length
    """
    patients = get_patients(masks_path)

    lengths = []
    for patient in patients:
        for slice in patient.cinemri_slices:
            mask_path = slice.build_path(masks_path)
            # Expect that contour does not change much across series, so taking the first frame
            mask = sitk.ReadImage(str(mask_path))
            if slice_complete_and_sagittal(mask):
                frame = sitk.GetArrayFromImage(mask)[0]
                x, y, _, _ = get_contour(frame)
                lengths.append(len(x))

    mean_length = np.mean(lengths)
    return mean_length

