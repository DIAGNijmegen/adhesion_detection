import os
import cv2
import SimpleITK as sitk

class CineMRISlice():
    def __init__(self, id, patient_id, examination_id):
        self.id = id
        self.patient_id = patient_id
        self.examination_id = examination_id

    def build_path(self, relative_path):
        return os.path.join(relative_path, self.patient_id, self.examination_id, self.id)


def extract_segmentation_data(archive_path,
                              destination_path,
                              images_folder="images",
                              segmentations_folder="cavity_segmentations",
                              target_images_folder="images",
                              target_segmentations_folder="masks"):

    # create target folders
    try:
        os.mkdir(destination_path)
    except FileExistsError:
        pass

    target_images_path = os.path.join(destination_path, target_images_folder)
    target_segmentations_path = os.path.join(destination_path, target_segmentations_folder)

    try:
        os.mkdir(target_images_path)
    except FileExistsError:
        pass

    try:
        os.mkdir(target_segmentations_path)
    except FileExistsError:
        pass

    # Paths
    images_path = os.path.join(archive_path, images_folder)
    segmentations_path = os.path.join(archive_path, segmentations_folder)

    # Patient ids
    patient_ids = [f.name for f in os.scandir(segmentations_path) if f.is_dir()]

    # Now get all examinations for each patient and create an array of Patients
    slices = []
    for patient_id in patient_ids:
        # Create patient folder in both images and masks target directories
        patient_images_path = os.path.join(target_images_path, patient_id)
        try:
            os.mkdir(patient_images_path)
        except FileExistsError:
            pass

        patient_segmentations_path = os.path.join(target_segmentations_path, patient_id)
        try:
            os.mkdir(patient_segmentations_path)
        except FileExistsError:
            pass

        # Find all examinations for a patient
        patient_path = os.path.join(segmentations_path, patient_id)
        examinations = [f.name for f in os.scandir(patient_path) if f.is_dir()]

        for examination_id in examinations:
            # Create examination folder in both images and masks target directories
            examination_images_path = os.path.join(patient_images_path, examination_id)
            try:
                os.mkdir(examination_images_path)
            except FileExistsError:
                pass

            examination_segmentations_path = os.path.join(patient_segmentations_path, examination_id)
            try:
                os.mkdir(examination_segmentations_path)
            except FileExistsError:
                pass

            # Find all segmented slices
            examination_path = os.path.join(patient_path, examination_id)
            slice_ids = [f for f in os.listdir(examination_path) if os.path.isfile(os.path.join(examination_path, f))]

            for slice_id in slice_ids:
                # TODO: see if there is loss in quality here
                # Or maybe just save as MHA?
                # read an image and extract the first frame
                scan_path = os.path.join(images_path, patient_id, examination_id, slice_id)
                image = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))[0]
                image_target_path = os.path.join(examination_images_path, os.path.splitext(slice_id)[0])
                cv2.imwrite("{}.png".format(image_target_path), image/image.max() * 255)

                # read a segmentation mask and extract the first frame
                segmentation_path = os.path.join(segmentations_path, patient_id, examination_id, slice_id)
                mask = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))[0]
                mask_target_path = os.path.join(examination_segmentations_path, os.path.splitext(slice_id)[0])
                cv2.imwrite("{}.png".format(mask_target_path), mask*255)

                slices.append(CineMRISlice(slice_id, patient_id, examination_id))
    
    return slices


if __name__ == '__main__':
    archive_path = "../../data/cinemri_mha/rijnstate"
    destination_path = "../../data/cinemri_mha/segmentation_subset"

    extract_segmentation_data(archive_path, destination_path)