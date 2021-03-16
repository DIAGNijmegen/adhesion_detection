import random
import numpy as np
import json
from pathlib import Path
import SimpleITK as sitk
from cinemri.utils import get_patients


class CineMRISlice:

    def __init__(self, file_name, patient_id, scan_id):
        self.file_name = file_name
        self.patient_id = patient_id
        self.scan_id = scan_id

    def build_path(self, relative_path):
        return Path(relative_path) / self.patient_id / self.scan_id / self.file_name


def get_patients_without_slices(archive_path,
                                images_folder="images"):
    """
    Finds patients who do not have any slices
    :param archive_path: a path to the full cine-MRI data archive
    :param images_folder: a name of the images folder in the archive
    :return: a list of patients without slices
    """

    patients = get_patients(archive_path / images_folder, with_scans_only=False)
    patients_without_slices = []
    for patient in patients:
        if len(patient.slices()) == 0:
            patients_without_slices.append(patient.id)

    return patients_without_slices


def train_test_split(archive_path,
                     split_destination,
                     images_folder="cavity_segmentations",
                     train_proportion=0.8):
    """
    Creates training/test split by patients
    :param archive_path: a path to the full cine-MRI data archive
    :param split_destination: a parth to save a json file with training/test split
    :param images_folder: a name of the images folder in the archive
    :param train_proportion: a share of the data to use for training
    :return: a tuple with a list of patients to use for training and  a list of patients to use for testing
    """

    patients = get_patients(archive_path / images_folder)
    random.shuffle(patients)
    train_size = round(len(patients) * train_proportion)

    train_patients = patients[:train_size]
    test_patients = patients[train_size:]

    train_patients_ids = [patient.id for patient in train_patients]
    test_patients_ids = [patient.id for patient in test_patients]
    split_json = {"train_patients": train_patients_ids, "test_patients_ids": test_patients_ids}

    dest_path = Path(split_destination)
    dest_path.mkdir(exist_ok=True)
    split_file_path = dest_path / "train_test_split.json"
    with open(split_file_path, "w") as f:
        json.dump(split_json, f)

    return train_patients, test_patients


def find_unique_shapes(archive_path, images_folder="images"):
    """
    Finds unique shapes of slices in the archive
    :param archive_path: a path to the full archive
    :param images_folder: a name of the images folder in the archive
    :return: a list of unique shapes
    """
    shapes = []

    patients = get_patients(archive_path / images_folder)
    for patient in patients:
        for scan_id, slices in patient.scans.items():
            for slice in slices:
                slice_image_path = archive_path / images_folder / patient.id / scan_id / slice
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_image_path)))[0]

                if not (image.shape in shapes):
                    shapes.append(image.shape)

    return shapes


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")

    train_test_split(archive_path, subset_path, train_proportion=1)

    """
    unique_shapes = find_unique_shapes(archive_path, "cavity_segmentations")
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


