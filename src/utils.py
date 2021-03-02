from pathlib import Path
import SimpleITK as sitk
from cinemri.utils import get_patients


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


# from the main archive
def find_unique_shapes(archive_path, images_folder="images"):
    shapes = []

    patients = get_patients(archive_path, images_folder)
    for patient in patients:
        for examination_id, slices in patient.examinations.items():
            for slice in slices:
                slice_image_path = archive_path / images_folder / patient.id / examination_id / slice
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_image_path)))[0]

                if image.shape == (192, 256):
                    print(slice_image_path)

                if not (image.shape in shapes):
                    shapes.append(image.shape)

    return shapes


if __name__ == '__main__':
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")
    pre_nnUNet_path = Path("../../data/cinemri_mha/pre_nnUNet_custom")

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
