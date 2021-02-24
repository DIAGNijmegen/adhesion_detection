import os
from torch.utils.data import Dataset
from utils import CineMRISlice
import SimpleITK as sitk
import matplotlib.pyplot as plt

# data set should not load the data in init function
# it should prepare prepare the way lo quickly access the data
# so the idea is to make a simple base class with little functionality and create subclasses for specific functionality

data_path = "../../data/cinemri_mha/rijnstate"

class SegmentationDatasetFull(Dataset):
    def __init__(self,
                 archive_path,
                 images_folder="images",
                 segmentations_folder="cavity_segmentations"):
        self.archive_path = archive_path
        self.images_folder = images_folder
        self.segmentations_folder = segmentations_folder

        # Paths
        self.images_path = os.path.join(archive_path, images_folder)
        self.segmentations_path = os.path.join(archive_path, segmentations_folder)
        self.__extract_segmented_slices__()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice = self.slices[idx]
        image_path = slice.build_path(self.images_path)
        segmentation_path = slice.build_path(self.segmentations_path)

        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))[0]
        mask = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))[0]

        return (image, mask)

    def __extract_segmented_slices__(self):
        # Patient ids
        self.patient_ids = [f.name for f in os.scandir(self.segmentations_path) if f.is_dir()]

        # Now get all examinations for each patient and create an array of Patients
        self.slices = []
        for patient_id in self.patient_ids:
            patient_segmentations_path = os.path.join(self.segmentations_path, patient_id)
            examinations = [f.name for f in os.scandir(patient_segmentations_path) if f.is_dir()]

            for examination_id in examinations:
                examination_path = os.path.join(patient_segmentations_path, examination_id)
                slice_ids = [f for f in os.listdir(examination_path) if
                             os.path.isfile(os.path.join(examination_path, f))]

                for slice_id in slice_ids:
                    self.slices.append(CineMRISlice(slice_id, patient_id, examination_id))


def main():
    dataset = SegmentationDatasetFull(data_path)
    for i in range(10):
        sample = dataset[i]

        plt.figure()
        plt.imshow(sample[0], cmap="gray")
        plt.imshow(sample[1], alpha=0.2)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()