#!/usr/local/bin/python3

import random
import numpy as np
import json
import pickle
from pathlib import Path
import SimpleITK as sitk
from .config import *
from .vs_definitions import VisceralSlide
from cinemri.utils import get_patients, get_image_orientation
from cinemri.definitions import Patient, CineMRISlice, Study

SAGITTAL_ORIENTATION = "ASL"


def slice_complete_and_sagittal(image):
    depth = image.GetDepth()
    orientation = get_image_orientation(image)
    return depth >= 30 and orientation == SAGITTAL_ORIENTATION


# Patients related information
def get_patients_ids_at_path(images_path):
    """Extract ids of patients found at a given path"""
    patients = get_patients(images_path)
    patient_ids = [patient.id for patient in patients]
    return patient_ids


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


# Data reading and writing
def full_ids_to_file(full_ids, output_file_path):
    with open(output_file_path, "w") as f:
        for full_id in full_ids:
            f.write(full_id + "\n")


def load_patients_ids(ids_file_path):
    """Loads patient ids from a file in which ids are separated by a new line"""
    with open(ids_file_path) as file:
        lines = file.readlines()
        patients_ids = [line.strip() for line in lines]

    return patients_ids


def write_slices_to_file(slices, file_path):
    """Writes fill ids of cine-MRI slices to file separated by a new line"""
    slices_full_ids = [slice.full_id for slice in slices]
    with open(file_path, "w") as file:
        for full_id in slices_full_ids:
            file.write(full_id + "\n")


def slices_from_full_ids_file(slices_full_ids_file_path):
    """Reads full ids from cine-MRI slices from file and creates CineMRISlice for each full id"""
    with open(slices_full_ids_file_path) as file:
        lines = file.readlines()
        slices_full_ids = [line.strip() for line in lines]

    slices = [CineMRISlice.from_full_id(full_id) for full_id in slices_full_ids]
    return slices


def patients_from_full_ids_file(slices_full_ids_file_path):
    """Reads full ids from cine-MRI slices from file and extracts unique patients which
    the corresponding cine-MRI slices belong to
    """
    with open(slices_full_ids_file_path) as file:
        lines = file.readlines()
        slices_full_ids = [line.strip() for line in lines]

    return patients_from_full_ids(slices_full_ids)


def patients_from_full_ids(slices_full_ids):
    """Extracts unique patients from the list of cine-MRI slices full ids"""
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
    """Extract full ids of cine-MRI slices that belong to specified list of patients"""
    slices_full_ids = []
    for patient in patients:
        slices_full_ids.extend([slice.full_id for slice in patient.cinemri_slices])

    return slices_full_ids


def patients_from_metadata(patients_metadata_path):
    """
    Load patients with all the extracted metadata from a metadata file

    Parameters
    ----------
    patients_metadata_path : Path
       A path to a file with patient metadata
    Returns
    -------
    patients : list of Patient
       A list of loaded patients
    """

    with open(patients_metadata_path) as f:
        patients_json = json.load(f)

    patients = [Patient.from_dict(patient_dict) for patient_dict in patients_json]
    return patients


def load_visceral_slides(visceral_slide_path):
    """
    Loads visceral slide data from disk
    Parameters
    ----------
    visceral_slide_path : Path
       A path to a folder that contains visceral slide data. The expected folder hierarchy is
       patient
          study1
             slice1
                 "visceral_slide.pkl"
             slice2
                 "visceral_slide.pkl"
             ...
             sliceN
                 "visceral_slide.pkl"
          study2
          ...
    Returns
    -------
    visceral_slides : list of VisceralSlide
       A list of visceral slides found at the path
    """
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
                if not visceral_slide_data_path.is_file():
                    continue
                with open(str(visceral_slide_data_path), "r+b") as file:
                    visceral_slide_data = pickle.load(file)

                visceral_slide = VisceralSlide(
                    patient_id, study_id, slice_id, visceral_slide_data
                )
                visceral_slides.append(visceral_slide)

    return visceral_slides


# Intervals and ranges handling
def binning_intervals(start=0, end=1, n=1000):
    """Splits the (start, end) range into n intervals and return the middle value of each interval"""
    intervals = np.linspace(start, end, n + 1)
    reference_vals = [(intervals[i] + intervals[i + 1]) / 2 for i in range(n)]
    return reference_vals


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


# Inspiration / expiration frames
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
