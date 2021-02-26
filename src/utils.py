import os
import numpy as np
import cv2
import SimpleITK as sitk
from cinemri.utils import get_patients, fill_contourous_masks


class CineMRISlice:
    def __init__(self, file_name, patient_id, examination_id):
        self.file_name = file_name
        self.patient_id = patient_id
        self.examination_id = examination_id

        self._image = None
        self._mask = None

    def build_path(self, relative_path):
        return os.path.join(relative_path, self.patient_id, self.examination_id, self.file_name)


def get_patients_without_slices(archive_path,
                                images_folder="images"):

    patients = get_patients(archive_path, images_folder=images_folder)
    patients_without_slices = []
    for patient in patients:
        if len(patient.slices()) == 0:
            patients_without_slices.append(patient.id)

    return patients_without_slices


# TODO: see if there is loss in quality when converting to .png
def save_frame(source_path, target_path,  slice_file, index=0):
    image = sitk.GetArrayFromImage(sitk.ReadImage(source_path))[index]
    image_target_path = os.path.join(target_path, os.path.splitext(slice_file)[0])
    # Save .npy
    np.save("{}.npy".format(image_target_path), image)
    # Save .png
    cv2.imwrite("{}.png".format(image_target_path), image / image.max() * 255)


def extract_segmentation_data(archive_path,
                              destination_path,
                              images_folder="images",
                              segmentations_folder="cavity_segmentations",
                              target_images_folder="images",
                              target_segmentations_folder="masks"):

    target_images_path = os.path.join(destination_path, target_images_folder)
    target_segmentations_path = os.path.join(destination_path, target_segmentations_folder)

    # create target folders
    try:
        os.mkdir(destination_path)
        os.mkdir(target_images_path)
        os.mkdir(target_segmentations_path)
    except FileExistsError:
        pass

    # Paths
    images_path = os.path.join(archive_path, images_folder)
    segmentations_path = os.path.join(archive_path, segmentations_folder)

    patients = get_patients(archive_path, images_folder=segmentations_folder)
    # Now get all examinations for each patient and create an array of Patients
    for patient in patients:
        if len(patient.slices()) == 0:
            continue

        # Create patient folder in both images and masks target directories
        patient_images_path = os.path.join(target_images_path, patient.id)
        patient_segmentations_path = os.path.join(target_segmentations_path, patient.id)

        try:
            os.mkdir(patient_images_path)
            os.mkdir(patient_segmentations_path)
        except FileExistsError:
            pass

        for (examination_id, slices) in patient.examinations.items():
            if len(slices) == 0:
                continue

            # Create examination folder in both images and masks target directories
            examination_images_path = os.path.join(patient_images_path, examination_id)
            examination_segmentations_path = os.path.join(patient_segmentations_path, examination_id)
            try:
                os.mkdir(examination_images_path)
                os.mkdir(examination_segmentations_path)
            except FileExistsError:
                pass

            for slice in slices:
                # read an image and extract the first frame
                slice_path = os.path.join(images_path, patient.id, examination_id, slice)
                save_frame(slice_path, examination_images_path, slice)

                # read a segmentation mask and extract the first frame
                segmentation_path = os.path.join(segmentations_path, patient.id, examination_id, slice)
                save_frame(segmentation_path, examination_segmentations_path, slice)



if __name__ == '__main__':
    archive_path = "../../data/cinemri_mha/rijnstate"

    fill_contourous_masks(archive_path)

    patients_without_slices = get_patients_without_slices(archive_path)
    print("Patients without slices")
    print(patients_without_slices)

    patients_without_segmented_slices = get_patients_without_slices(archive_path, images_folder="cavity_segmentations")
    print("Patients without segmented slices")
    print(patients_without_segmented_slices)

    #extract_segmentation_data(archive_path, destination_path)