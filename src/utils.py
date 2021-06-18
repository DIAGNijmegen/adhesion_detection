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
from config import TRAIN_TEST_SPLIT_FILE_NAME, TRAIN_PATIENTS_KEY, TEST_PATIENTS_KEY, METADATA_FOLDER,\
    NEGATIVE_PATIENTS_FILE_NAME, PATIENTS_METADATA_FILE_NAME, NEW_REPORT_FILE_NAME, NEGATIVE_SLICES_FILE_NAME, IMAGES_FOLDER, \
    SEPARATOR, VISCERAL_SLIDE_FILE
from cinemri.config import ARCHIVE_PATH
from cinemri.utils import get_patients, get_image_orientation
from cinemri.contour import get_contour
from cinemri.definitions import Patient, CineMRISlice
import shutil

ADHESIONS_KEY_NEW = "adhesion"

# TODO: move
class VisceralSlide:
    """An object representing visceral slide for a Cine-MRI slice
    """
    def __init__(self, patient_id, study_id, slice_id, visceral_slide_data):
        self.patient_id = patient_id
        self.study_id = study_id
        self.slice_id = slice_id
        self.full_id = SEPARATOR.join([patient_id, study_id, slice_id])
        self.x = visceral_slide_data["x"]
        self.y = visceral_slide_data["y"]
        self.values = visceral_slide_data["slide"]

    def build_path(self, relative_path, extension=".mha"):
        return Path(relative_path) / self.patient_id / self.study_id / (self.slice_id + extension)


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


# TODO: document
def load_negative_patients_ids(report_path, new_only=True):

    with open(report_path) as f:
        report_data = json.load(f)

    # Filter out ids related to the old dataset
    if new_only:
        old_nums = range(1, 65)
        old_ids = ["R{:0>3d}".format(num) for num in old_nums]
    else:
        old_ids = []

    negative_patients_ids = []
    for patient_id, report in report_data.items():
        if report[ADHESIONS_KEY_NEW] == 0 and patient_id not in old_ids:
            negative_patients_ids.append(patient_id)

    return negative_patients_ids


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



# Save as metadata file
def sample_slices(report_path, patients_metadata_path, metadata_path, patients_num=63, slices_per_patient=1):

    negative_ids = load_negative_patients_ids(report_path)

    # Load patients with all the extracted metadata
    with open(patients_metadata_path) as f:
        patients_json = json.load(f)

    patients = [Patient.from_dict(patient_dict) for patient_dict in patients_json]

    # Leave only negative patients
    negative_patients = [patient for patient in patients if patient.id in negative_ids]
    sampled_patients_ids = []
    sampled_slices = []

    while len(sampled_patients_ids) < patients_num or len(negative_patients) == 0:
        # Sample patient
        patient = np.random.choice(negative_patients, 1)[0]
        # Sample slice
        suitable_slices = [slice for slice in patient.cinemri_slices if slice.vs_suitable]

        if len(suitable_slices) > 0:
            n = min(len(suitable_slices), slices_per_patient)
            slices = np.random.choice(suitable_slices, n)
            sampled_slices += slices.tolist()
            sampled_patients_ids.append(patient.id)

        negative_patients.remove(patient)

    # Write full ids of sampled slices to file
    output_file_path = metadata_path / NEGATIVE_SLICES_FILE_NAME
    write_slices_to_file(sampled_slices, output_file_path)

    return sampled_slices


def sample_slices_detection(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_path', type=str, required=True,
                        help="a path to a file with patients ids to sample slices from")
    parser.add_argument('--patients_metadata_path', type=str, required=True, help="a path to the cine-MRI archive")
    parser.add_argument('--metadata_path', type=str, required=True,
                        help="a path to the metadata folder of the cine-MRI archive")
    parser.add_argument('--patients_num', type=str, default=63,
                        help="a number of different patients to sample")
    parser.add_argument('--slices_num', type=str, default=1,
                        help="a number of slices per patient to sample")

    args = parser.parse_args(argv)
    report_path = Path(args.report_path)
    patients_metadata_path = Path(args.patients_metadata_path)
    metadata_path = Path(args.metadata_path)
    patients_num = args.patients_num
    slices_num = args.slices_num

    sample_slices(report_path, patients_metadata_path, metadata_path, patients_num, slices_num)


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


def slices_from_full_ids_file(slices_full_ids_file_path):
    with open(slices_full_ids_file_path) as file:
        lines = file.readlines()
        slices_full_ids = [line.strip() for line in lines]

    slices = [CineMRISlice.from_full_id(full_id) for full_id in slices_full_ids]
    return slices


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
    parser.add_argument('--positive_file', type=str, required=True, help="a path to a file with fill ids of positive slices")
    parser.add_argument('--negative_file', type=str, required=True, help="a path to a file with fill ids of negative slices")
    parser.add_argument('--images', type=str, required=True, help="a path to image folder in the cine-MRI archive")
    parser.add_argument('--target_folder', type=str, required=True, help="a path to a folder to place the detection subset")

    args = parser.parse_args(argv)

    positive_file_path = Path(args.positive_file)
    negative_file_path = Path(args.negative_file)
    images_path = Path(args.images)
    target_path = Path(args.target_folder) / IMAGES_FOLDER
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



def test():
    archive_path = Path(ARCHIVE_PATH)

    metadata_path = archive_path / METADATA_FOLDER
    report_path = metadata_path / NEW_REPORT_FILE_NAME
    patients_metadata = metadata_path / PATIENTS_METADATA_FILE_NAME
    
    #slices = sample_slices(report_path, patients_metadata, metadata_path)

    full_ids = Path("detection_ids.txt")
    added_ids = Path("ids.txt")

    patients_ids = load_patients_ids(added_ids)
    slices = slices_from_full_ids_file(full_ids)

    missed_slices = [slice for slice in slices if slice.patient_id not in patients_ids]
    write_slices_to_file(missed_slices, Path("missed_detection_ids.txt"))




    print("done")
    #full_segmentation_path = archive_path / "full_segmentation"
    #contour_stat(full_segmentation_path)
    #train_test_split(archive_path, subset_path, train_proportion=1)



if __name__ == '__main__':

    np.random.seed(99)
    random.seed(99)

    # Very first argument determines action
    actions = {
        "extract_detection_data": extract_detection_data,
        "sample_slices": sample_slices_detection
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: registration ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])

