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
from cinemri.definitions import Patient, CineMRISlice, Study, AnatomicalPlane, CineMRIMotionType, CineMRISlicePos
import shutil




# TODO: moveis_slice_vs_suitable
class VisceralSlide:
    """An object representing visceral slide for a Cine-MRI slice
    """

    def __init__(self, patient_id, study_id, slice_id, visceral_slide_data):
        self.patient_id = patient_id
        self.study_id = study_id
        self.slice_id = slice_id
        self.full_id = SEPARATOR.join([patient_id, study_id, slice_id])
        self.x = np.array(visceral_slide_data["x"])
        self.y = np.array(visceral_slide_data["y"])
        self.values = np.array(visceral_slide_data["slide"])
        self.origin_x = np.min(self.x)
        self.origin_y = np.min(self.y)
        self.width = np.max(self.x) - self.origin_x
        self.height = np.max(self.y) - self.origin_y
        self.middle_x = self.origin_x + round(self.width / 2)
        self.middle_y = self.origin_y + round(self.height / 2)

        self.__top_coords = None
        self.__bottom_coords = None

        self.__top_left_coords = None
        self.__top_right_coords = None
        self.__bottom_left_coords = None
        self.__bottom_right_coords = None

        self.__bottom_left_point = None
        self.__top_left_point = None
        self.__bottom_right_point = None
        self.__top_right_point = None

    @property
    def top_coords(self):
        if self.__top_coords is None:
            coords = np.column_stack((self.x, self.y))
            self.__top_coords = np.array([coord for coord in coords if coord[1] < self.middle_y])

        return self.__top_coords

    @property
    def bottom_coords(self):
        if self.__bottom_coords is None:
            coords = np.column_stack((self.x, self.y))
            self.__bottom_coords = np.array([coord for coord in coords if coord[1] >= self.middle_y])

        return self.__bottom_coords
    
    @property
    def top_middle_x(self):
        top_coords_x = self.top_coords[:, 0]
        return top_coords_x.min() + (top_coords_x.max() - top_coords_x.min()) / 2

    @property
    def bottom_middle_x(self):
        bottom_coords_x = self.bottom_coords[:, 0]
        return bottom_coords_x.min() + (bottom_coords_x.max() - bottom_coords_x.min()) / 2

    @property
    def top_left_coords(self):
        if self.__top_left_coords is None:
            top_left_coords = np.array([coord for coord in self.top_coords if coord[0] < self.top_middle_x])
            self.__top_left_coords = top_left_coords[:, 0], top_left_coords[:, 1]

        return self.__top_left_coords

    @property
    def top_right_coords(self):
        if self.__top_right_coords is None:
            top_right_coords = np.array([coord for coord in self.top_coords if coord[0] >= self.top_middle_x])
            self.__top_right_coords = top_right_coords[:, 0], top_right_coords[:, 1]

        return self.__top_right_coords

    @property
    def bottom_left_coords(self):
        if self.__bottom_left_coords is None:
            bottom_left_coords = np.array([coord for coord in self.bottom_coords if coord[0] < self.bottom_middle_x])
            self.__bottom_left_coords = bottom_left_coords[:, 0], bottom_left_coords[:, 1]

        return self.__bottom_left_coords

    @property
    def bottom_right_coords(self):
        if self.__bottom_right_coords is None:
            bottom_right_coords = np.array([coord for coord in self.bottom_coords if coord[0] >= self.bottom_middle_x])
            self.__bottom_right_coords = bottom_right_coords[:, 0], bottom_right_coords[:, 1]

        return self.__bottom_right_coords

    @property
    def bottom_left_point(self):
        x, y = self.bottom_left_coords
        x, y = x.astype(np.float64), y.astype(np.float64)
        bottom_left_x = x.min()
        bottom_left_y = y.max()

        diff = np.sqrt((x - bottom_left_x) ** 2 + (y - bottom_left_y) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    @property
    def top_left_point(self):
        x, y = self.top_left_coords
        x, y = x.astype(np.float64), y.astype(np.float64)

        diff = np.sqrt((x - x.min()) ** 2 + (y - y.min()) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    @property
    def bottom_right_point(self):
        x, y = self.bottom_right_coords
        x, y = x.astype(np.float64), y.astype(np.float64)

        diff = np.sqrt((x - x.max()) ** 2 + (y - y.max()) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    @property
    def top_right_point(self):
        x, y = self.top_right_coords
        x, y = x.astype(np.float64), y.astype(np.float64)

        diff = np.sqrt((x - x.max()) ** 2 + (y - y.min()) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

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



def test():
    archive_path = ARCHIVE_PATH

    metadata_path = archive_path / METADATA_FOLDER
    report_path = metadata_path / REPORT_FILE_NAME
    patients_metadata = metadata_path / PATIENTS_METADATA_FILE_NAME
    mapping_path = archive_path / METADATA_FOLDER / PATIENTS_MAPPING_FILE_NAME
    patients = patients_from_metadata(patients_metadata)
    max_stud_num = 0
    for patient in patients:
        max_stud_num = max(max_stud_num, len(patient.studies))
        
    print("Maximum number of studies {}".format(max_stud_num))

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

    # full_segmentation_path = archive_path / "full_segmentation"
    # contour_stat(full_segmentation_path)
    # train_test_split(archive_path, subset_path, train_proportion=1)


if __name__ == '__main__':
    #test()

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
