#!/usr/local/bin/python3

# TDOD: create separate module with functions for dataset stats

import sys
import random
import numpy as np
import json
import argparse
import pickle
from pathlib import Path
import SimpleITK as sitk
from config import *
from cinemri.config import ARCHIVE_PATH
from cinemri.utils import get_patients, get_image_orientation
from cinemri.contour import get_contour
from cinemri.definitions import Patient, CineMRISlice, Study
import shutil
import matplotlib.pyplot as plt
from vs_definitions import VisceralSlide

# Splits the (start, end) range into n intervals and return the middle value of each interval
def binning_intervals(start=0, end=1, n=1000):
    intervals = np.linspace(start, end, n + 1)
    reference_vals = [(intervals[i] + intervals[i + 1]) / 2 for i in range(n)]
    return reference_vals


def get_patients_without_slices(images_path):
    """Finds patients who do not have any slices

    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images

    Returns
    -------
    list of Patient
       A list of patients without slices

    """

    patients = get_patients(images_path, with_slices_only=False)
    patients_without_slices = []
    for patient in patients:
        if len(patient.cinemri_slices()) == 0:
            patients_without_slices.append(patient.id)

    return patients_without_slices

# TODO: probably handing of pos/neg
def get_segm_patients_ids(images_path):
    patients = get_patients(images_path)
    patient_ids = [patient.id for patient in patients]
    return patient_ids


def train_test_split(images_path,
                     split_destination,
                     train_proportion=0.8):
    """Creates training/test split by patients

    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images
    split_destination : Path
       A path to save a json file with training/test split
    train_proportion : float, default=0.8
       A share of the data to use for training

    Returns
    -------
    tuple of list of string
       A tuple with a list of patients ids to use for training and a list of patients ids to use for testing
    """

    patients = get_patients(images_path)
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


def find_unique_shapes(images_path):
    """Finds unique shapes of slices in the archive

    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images
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


def select_segmentation_patients_subset(images_path, target_path, n=10):
    """
    Randomly samples a subset of patient for which to perform abdominal cavity segmentation
    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images
    target_path : Path
       A path where to create folders for selected patients
    n : int, default=10
       A sample size
    """

    target_path.mkdir(parents=True, exist_ok=True)

    patients = get_patients(images_path)
    patients_ids = [patient.id for patient in patients]

    ids_to_segm = random.sample(patients_ids, n)
    print("Patients in the sampled segmentation subset: {}".format(ids_to_segm))

    patients_subset = [patient for patient in patients if patient.id in ids_to_segm]
    for patient in patients_subset:
        patient.build_path(target_path).mkdir()
        for study in patient.studies:
            study.build_path(target_path).mkdir()


def load_patients_ids(ids_file_path):
    with open(ids_file_path) as file:
        lines = file.readlines()
        patients_ids = [line.strip() for line in lines]

    return patients_ids


def write_slices_to_file(slices, file_path):
    slices_full_ids = [slice.full_id for slice in slices]
    with open(file_path, "w") as file:
        for full_id in slices_full_ids:
            file.write(full_id + "\n")


def get_vs_range(visceral_slides, negative_vs_needed):
    """Returns visceral slide range with excluded outliers
    """

    # Statistics useful for prediction
    all_vs_values = []
    for visceral_slide in visceral_slides:
        all_vs_values.extend(visceral_slide.values)

    vs_abs_min = np.min(all_vs_values)
    vs_abs_max = np.max(all_vs_values)
    print("VS minumum : {}".format(vs_abs_min))
    print("VS maximum : {}".format(vs_abs_max))

    vs_q1 = np.quantile(all_vs_values, 0.25)
    vs_q3 = np.quantile(all_vs_values, 0.75)
    vs_iqr = vs_q3 - vs_q1
    vs_min = min(vs_abs_min, vs_q1 - 1.5 * vs_iqr)
    vs_max = min(vs_abs_max, vs_q3 + 1.5 * vs_iqr)

    print("VS minumum, outliers removed range : {}".format(vs_min))
    print("VS maximum, outliers removed range : {}".format(vs_max))

    return (-np.inf, 0) if negative_vs_needed else (vs_min, vs_max)

def get_avg_contour_size(visceral_slides):
    widths = []
    heights = []

    for vs in visceral_slides:
        widths.append(vs.width)
        heights.append(vs.height)

    return round(np.mean(widths)), round(np.mean(heights))


def bb_size_stat(annotations, is_median=False):
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

    average_size = (np.median(widths), np.median(heights)) if is_median else (np.mean(widths), np.mean(heights))
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


def slices_from_full_ids_file(slices_full_ids_file_path):
    with open(slices_full_ids_file_path) as file:
        lines = file.readlines()
        slices_full_ids = [line.strip() for line in lines]

    slices = [CineMRISlice.from_full_id(full_id) for full_id in slices_full_ids]
    return slices


def patients_from_full_ids_file(slices_full_ids_file_path):
    with open(slices_full_ids_file_path) as file:
        lines = file.readlines()
        slices_full_ids = [line.strip() for line in lines]

    return patients_from_full_ids(slices_full_ids)


def patients_from_full_ids(slices_full_ids):
    slices_id_chunks = [slice_full_id.split("_") for slice_full_id in slices_full_ids]
    slices_id_chunks = np.array(slices_id_chunks)

    patient_ids = np.unique(slices_id_chunks[:, 0])
    patients = []
    for patient_id in patient_ids:
        patient = Patient(patient_id)
        patient_records = slices_id_chunks[slices_id_chunks[:, 0] == patient_id]
        studies_ids = np.unique(patient_records[:, 1])

        for study_id in studies_ids:
            study = Study(study_id, patient_id=patient_id)
            study_records = patient_records[patient_records[:, 1] == study_id]

            for _, _, slice_id in study_records:
                study.add_slice(CineMRISlice(slice_id, patient_id, study_id))

            patient.add_study(study)

        patients.append(patient)

    return patients


def slices_full_ids_from_patients(patients):
    slices_full_ids = []
    for patient in patients:
        slices_full_ids.extend([slice.full_id for slice in patient.cinemri_slices])

    return slices_full_ids


def extract_detection_dataset(slices, images_folder, target_folder):
    # Copy slices to a new location
    for slice in slices:
        study_dir = target_folder / slice.patient_id / slice.study_id
        study_dir.mkdir(exist_ok=True, parents=True)
        slice_path = slice.build_path(images_folder)
        slice_target_path = slice.build_path(target_folder)
        shutil.copyfile(slice_path, slice_target_path)


def extract_detection_data(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_file', type=str, required=True,
                        help="a path to a file with fill ids of positive slices")
    parser.add_argument('--negative_file', type=str, required=True,
                        help="a path to a file with fill ids of negative slices")
    parser.add_argument('--images', type=str, required=True, help="a path to image folder in the cine-MRI archive")
    parser.add_argument('--target_folder', type=str, required=True,
                        help="a path to a folder to place the detection subset")

    args = parser.parse_args(argv)

    positive_file_path = Path(args.positive_file)
    negative_file_path = Path(args.negative_file)
    images_path = Path(args.images)
    target_path = Path(args.target_folder)
    target_path.mkdir(parents=True)

    positive_slices = slices_from_full_ids_file(positive_file_path)
    negative_slices = slices_from_full_ids_file(negative_file_path)
    slices = positive_slices + negative_slices
    extract_detection_dataset(slices, images_path, target_path)


def contour_stat(images_path):
    patients = get_patients(images_path)

    lengths = []
    for patient in patients:
        for slice in patient.cinemri_slices:
            slice_path = slice.build_path(images_path)
            # Expect that contour does not change much across series, so taking the first frame
            slice_image = sitk.ReadImage(str(slice_path))
            depth = slice_image.GetDepth()
            orientation = get_image_orientation(slice_image)
            if depth >= 30 and orientation == "ASL":
                frame = sitk.GetArrayFromImage(slice_image)[0]
                x, y, _, _ = get_contour(frame)
                lengths.append(len(x))

    mean_length = np.mean(lengths)
    print("Average contour length {}".format(mean_length))
    return mean_length


def load_visceral_slides(visceral_slide_path):
    patient_ids = [f.name for f in visceral_slide_path.iterdir() if f.is_dir()]

    visceral_slides = []
    for patient_id in patient_ids:
        patient_path = visceral_slide_path / patient_id
        studies = [f.name for f in patient_path.iterdir() if f.is_dir()]

        for study_id in studies:
            study_path = patient_path / study_id
            slices = [f.name for f in study_path.iterdir() if f.is_dir()]

            for slice_id in slices:
                visceral_slide_data_path = study_path / slice_id / VISCERAL_SLIDE_FILE
                with open(str(visceral_slide_data_path), "r+b") as file:
                    visceral_slide_data = pickle.load(file)

                visceral_slide = VisceralSlide(patient_id, study_id, slice_id, visceral_slide_data)
                visceral_slides.append(visceral_slide)

    return visceral_slides


def patients_from_metadata(patients_metadata_path):

    # Load patients with all the extracted metadata
    with open(patients_metadata_path) as f:
        patients_json = json.load(f)

    patients = [Patient.from_dict(patient_dict) for patient_dict in patients_json]
    return patients


# TODO: should be wrapped into try/cath
def get_insp_exp_indices(slice, inspexp_data):
    """
    Loads indexes of inspiration and expiration frames for the specified cine-MRI slice
    Parameters
    ----------
    slice: CineMRISlice
        A cine-MRI slice for which to extract inspiration and expiration frames
    inspexp_data : dict
        A dictionary with inspiration / expiration frames data

    Returns
    -------
    insp_ind, exp_ind : ndarray
        The inspiration and expiration frames indexes
    """

    patient_data = inspexp_data[slice.patient_id]
    study_data = patient_data[slice.study_id]
    inspexp_frames = study_data[slice.id]
    insp_ind = inspexp_frames[0]
    exp_ind = inspexp_frames[1]
    return insp_ind, exp_ind


# TODO: should be wrapped into try/cath
def get_inspexp_frames(slice, inspexp_data, images_path):
    """
    Loads inspiration and expiration frames for the specified cine-MRI slice
    Parameters
    ----------
    slice: CineMRISlice
        A cine-MRI slice for which to extract inspiration and expiration frames
    inspexp_data : dict
       A dictionary with inspiration / expiration frames data
    images_path : Path
       A path to the image folder in cine-MRI archive

    Returns
    -------
    insp_frame, exp_frame : ndarray
       The inspiration and expiration frames
    """

    insp_ind, exp_ind = get_insp_exp_indices(slice, inspexp_data)

    # Load the expiration frame (visceral slide is computed for the expiration frame)
    slice_path = slice.build_path(images_path)
    slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
    insp_frame = slice_array[insp_ind]
    exp_frame = slice_array[exp_ind]
    return insp_frame, exp_frame


def get_insp_exp_frames_and_masks(slice, inspexp_data, images_path, masks_path):
    """
    Loads inspiration and expiration frames of the slice and the corresponding mask
    Parameters
    ----------
    slice : CineMRISlice
        A cine-MRI slice for which to extract inspiration and expiration frames
    inspexp_data : dict
       A dictionary with inspiration / expiration frames data
    images_path, masks_path : Path
       Paths to images and masks

    Returns
    -------
    insp_frame, insp_mask, exp_frame, exp_mask : ndarray
       Inspiration frame and mask, expiration frame and mask

    """

    insp_ind, exp_ind = get_insp_exp_indices(slice, inspexp_data)

    # load image
    slice_path = slice.build_path(images_path)
    slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
    insp_frame = slice_array[insp_ind].astype(np.uint32)
    exp_frame = slice_array[exp_ind].astype(np.uint32)

    # load mask
    mask_path = slice.build_path(masks_path)
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    insp_mask = mask_array[insp_ind]
    exp_mask = mask_array[exp_ind]

    return insp_frame, insp_mask, exp_frame, exp_mask


def full_ids_to_file(full_ids, output_file_path):
    with open(output_file_path, "w") as f:
        for full_id in full_ids:
            f.write(full_id + "\n")


def test():
    archive_path = Path(ARCHIVE_PATH)

    metadata_path = archive_path / METADATA_FOLDER
    report_path = metadata_path / REPORT_FILE_NAME
    patients_metadata = metadata_path / PATIENTS_METADATA_FILE_NAME
    mapping_path = archive_path / METADATA_FOLDER / PATIENTS_MAPPING_FILE_NAME

    detection_path = Path(DETECTION_PATH) / IMAGES_FOLDER / TEST_FOLDER

    patients = get_patients(detection_path)
    slices_full_ids = slices_full_ids_from_patients(patients)
    full_ids_to_file(slices_full_ids, Path("test_full_ids.txt"))

    
    """
    patients = patients_from_metadata(patients_metadata)
    max_stud_num = 0
    for patient in patients:
        max_stud_num = max(max_stud_num, len(patient.studies))
        
    print("Maximum number of studies {}".format(max_stud_num))
    """

    """
    patients = patients_from_metadata("patients.json")

    print(len(patients))

    patients_few_studies = [patient for patient in patients if len(patient.studies) > 1]
    patients_few_studies_ids = [(p.id, len(p.studies)) for p in patients_few_studies]
    print(len(patients_few_studies_ids))
    print(patients_few_studies_ids)

    studies = []
    for p in patients:
        studies += p.studies

    studies_no_date = [s for s in studies if s.date is None]
    studies_no_date_ids = [(s.patient_id, s.id, len(s.slices)) for s in studies_no_date]
    print(len(studies_no_date_ids))
    print(studies_no_date_ids)
    print("done")
    """

    # full_segmentation_path = archive_path / FULL_SEGMENTATION_FOLDER
    # contour_stat(full_segmentation_path)
    # train_test_split(archive_path, subset_path, train_proportion=1)


if __name__ == '__main__':
    test()
    
    """
    np.random.seed(99)
    random.seed(99)

    # Very first argument determines action
    actions = {
        "extract_detection_data": extract_detection_data,
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: registration ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
    """
