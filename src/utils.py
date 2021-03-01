from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk
from cinemri.utils import get_patients, fill_contourous_masks

class CineMRISlice:
    def __init__(self, file_name, patient_id, examination_id):
        self.file_name = file_name
        self.patient_id = patient_id
        self.examination_id = examination_id

    def build_path(self, relative_path):
        return Path(relative_path) / self.patient_id / self.examination_id / self.file_name


def get_patients_without_slices(archive_path,
                                images_folder="images"):

    patients = get_patients(archive_path, images_folder=images_folder, with_scans_only=False)
    patients_without_slices = []
    for patient in patients:
        if len(patient.slices()) == 0:
            patients_without_slices.append(patient.id)

    return patients_without_slices


# TODO: see if there is loss in quality when converting to .png
def save_frame(source_path, target_path, index=0):
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(source_path)))[index]

    image_target_path = Path(target_path) / Path(source_path).stem
    # Save .npy
    np.save(image_target_path.with_suffix(".npy"), image)
    # Save .png
    cv2.imwrite(str(image_target_path.with_suffix(".png")), image / image.max() * 255)


def extract_segmentation_data(archive_path,
                              destination_path,
                              images_folder="images",
                              segmentations_folder="cavity_segmentations",
                              target_images_folder="images",
                              target_segmentations_folder="masks"):

    # create target paths and folders
    destination_path = Path(destination_path)
    destination_path.mkdir(exist_ok=True)

    target_images_path = destination_path / target_images_folder
    target_images_path.mkdir(exist_ok=True)

    target_segmentations_path = destination_path / target_segmentations_folder
    target_segmentations_path.mkdir(exist_ok=True)

    patients = get_patients(archive_path, images_folder=segmentations_folder)
    # Now get all examinations for each patient and create an array of Patients
    for patient in patients:
        if len(patient.slices()) == 0:
            continue

        # Create patient folder in both images and masks target directories
        patient_images_path = target_images_path / patient.id
        patient_images_path.mkdir(exist_ok=True)

        patient_segmentations_path = target_segmentations_path / patient.id
        patient_segmentations_path.mkdir(exist_ok=True)

        for (examination_id, slices) in patient.examinations.items():
            if len(slices) == 0:
                continue

            # Create examination folder in both images and masks target directories
            examination_images_path = patient_images_path / examination_id
            examination_images_path.mkdir(exist_ok=True)

            examination_segmentations_path = patient_segmentations_path / examination_id
            examination_segmentations_path.mkdir(exist_ok=True)

            for slice in slices:
                # read an image and extract the first frame
                slice_path = Path(archive_path) / images_folder / patient.id / examination_id / slice
                save_frame(slice_path, examination_images_path)

                # read a segmentation mask and extract the first frame
                segmentation_path = Path(archive_path) / segmentations_folder / patient.id / examination_id / slice
                save_frame(segmentation_path, examination_segmentations_path)


if __name__ == '__main__':
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")

    fill_contourous_masks(archive_path)

    patients_without_slices = get_patients_without_slices(archive_path)
    print("Patients without slices")
    print(patients_without_slices)

    patients_without_segmented_slices = get_patients_without_slices(archive_path, images_folder="cavity_segmentations")
    print("Patients without segmented slices")
    print(patients_without_segmented_slices)

    #extract_segmentation_data(archive_path, subset_path)