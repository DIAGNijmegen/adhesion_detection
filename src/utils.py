import random
import numpy as np
import json
from pathlib import Path
import SimpleITK as sitk
from config import TRAIN_TEST_SPLIT_FILE_NAME, TRAIN_PATIENTS_KEY, TEST_PATIENTS_KEY
from cinemri.utils import get_patients


def get_patients_without_slices(archive_path,
                                images_folder="images"):
    """Finds patients who do not have any slices

    Parameters
    ----------
    archive_path : Path
       A path to the full cine-MRI data archive
    images_folder : str, default="images"
       A name of the images folder in the archive

    Returns
    -------
    list of Patient
       A list of patients without slices

    """

    patients = get_patients(archive_path, images_folder=images_folder, with_scans_only=False)
    patients_without_slices = []
    for patient in patients:
        if len(patient.slices()) == 0:
            patients_without_slices.append(patient.id)

    return patients_without_slices


def train_test_split(archive_path,
                     split_destination,
                     images_folder="cavity_segmentations",
                     train_proportion=0.8):
    """Creates training/test split by patients

    Parameters
    ----------
    archive_path : Path
       A path to the full cine-MRI data archive
    split_destination : Path
       A path to save a json file with training/test split
    images_folder : str, default=="cavity_segmentations"
       A name of the images folder in the archive
    train_proportion : float, default=0.8
       A share of the data to use for training

    Returns
    -------
    tuple of list of string
       A tuple with a list of patients ids to use for training and a list of patients ids to use for testing
    """

    patients = get_patients(archive_path, images_folder=images_folder)
    random.shuffle(patients)
    train_size = round(len(patients) * train_proportion)

    train_patients = patients[:train_size]
    test_patients = patients[train_size:]

    train_patients_ids = [patient.id for patient in train_patients]
    test_patients_ids = [patient.id for patient in test_patients]
    split_json = {TRAIN_PATIENTS_KEY: train_patients_ids, TEST_PATIENTS_KEY: test_patients_ids}

    split_destination.mkdir(exist_ok=True)
    split_file_path = split_destination / TRAIN_TEST_SPLIT_FILE_NAME
    with open(split_file_path, "w") as f:
        json.dump(split_json, f)

    return train_patients, test_patients


def find_unique_shapes(archive_path, images_folder="images"):
    """Finds unique shapes of slices in the archive

    Parameters
    ----------
    archive_path : Path
       A path to a full archive
    images_folder : str, default="images"
       A list of unique slices shapes
    """
    shapes = []

    patients = get_patients(archive_path, images_folder=images_folder)
    for patient in patients:

        for cinemri_slice in patient.cinemri_slices:
            slice_image_path = cinemri_slice.build_path(archive_path / images_folder)
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_image_path)))[0]

            if not (image.shape in shapes):
                shapes.append(image.shape)

    return shapes


def interval_overlap(interval_a, interval_b):
    """
    Calculates a length of an overlap of two intervals
    Parameters
    ----------
    interval_a, interval_b : list of float
       Start and end coordinates of the interval [min, max]

    Returns
    -------
       A length of an overlap between intervals
    """
    a_x_min, a_x_max = interval_a
    b_x_min, b_x_max = interval_b

    if b_x_min < a_x_min:
        return 0 if b_x_max < a_x_min else min(a_x_max, b_x_max) - a_x_min
    else:
        return 0 if a_x_max < b_x_min else min(a_x_max, b_x_max) - b_x_min


def select_segmentation_patients_subset(archive_path, target_path, n=10):
    """
    Randomly samples a subset of patient for which to perform abdominal cavity segmentation
    Parameters
    ----------
    archive_path : Path
       A path to the full cine-MRI data archive
    target_path : Path
       A path where to create folders for selected patients
    n : int, default=10
       A sample size
    """

    target_path.mkdir(parents=True, exist_ok=True)

    patients = get_patients(archive_path)
    patients_ids = [patient.id for patient in patients]

    ids_to_segm = random.sample(patients_ids, n)
    print("Patients in the sampled segmentation subset: {}".format(ids_to_segm))

    patients_subset = [patient for patient in patients if patient.id in ids_to_segm]
    for patient in patients_subset:
        patient_dir = target_path / patient.id
        patient_dir.mkdir()
        for scan_id in patient.scan_ids:
            scan_dir = patient_dir / scan_id
            scan_dir.mkdir()


def average_bb_size(annotations):
    """
    Computes an average adhesion boundig box size

    Parameters
    ----------
    annotations : list of AdhesionAnnotation
       List of annotations for which an average size of a bounding box should be determined

    Returns
    -------
    width, height : int
       A tuple of average width and height in adhesion annotations with bounding boxes
    """

    widths = []
    heights = []

    for annotation in annotations:
        for adhesion in annotation.adhesions:
            widths.append(adhesion.width)
            heights.append(adhesion.height)

    return np.mean(widths), np.mean(heights)


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

    print("Smallest annotation, x_o: {}, y_o: {}, width: {}, height: {}".format(smallest_bb.origin_x, smallest_bb.origin_y, smallest_bb.width, smallest_bb.height))
    print("Largest annotation, width: {}, height: {}".format(largest_bb.width, largest_bb.height))


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")

    #train_test_split(archive_path, subset_path, train_proportion=1)
    """
    unique_shapes = find_unique_shapes(archive_path, "images")
    print("Unique scan dimensions in the dataset")
    print(unique_shapes)
    """

    """
    patients_without_slices = get_patients_without_slices(archive_path)
    print("Patients without slices")
    print(patients_without_slices)

    patients_without_segmented_slices = get_patients_without_slices(archive_path, images_folder="cavity_segmentations")
    print("Patients without segmented slices")
    print(patients_without_segmented_slices)
    """


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(99)
    test()


