import os
import numpy as np
from torch.utils.data import Dataset
from utils import CineMRISlice
import SimpleITK as sitk
import matplotlib.pyplot as plt


class SegmentationDatasetFull(Dataset):
    def __init__(self,
                 archive_path,
                 images_folder="images",
                 segmentations_folder="cavity_segmentations"):
        # Paths
        self.images_path = os.path.join(archive_path, images_folder)
        self.segmentations_path = os.path.join(archive_path, segmentations_folder)

        # Caches
        self.images = {}
        self.masks = {}

        self.__extract_segmented_slices__()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice = self.slices[idx]

        if slice.file_name in self.images:
            image = self.images[slice.file_name]
        else:
            image_path = slice.build_path(self.images_path)
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))[0]
            self.images[slice.file_name] = image

        if slice.file_name in self.masks:
            mask = self.masks[slice.file_name]
        else:
            segmentation_path = slice.build_path(self.segmentations_path)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))[0]
            self.masks[slice.file_name] = mask

        return image, mask

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
                slices = [f for f in os.listdir(examination_path) if
                             os.path.isfile(os.path.join(examination_path, f))]

                for slice in slices:
                    self.slices.append(CineMRISlice(slice, patient_id, examination_id))


class SegmentationDataset(Dataset):
    def __init__(self,
                 archive_path,
                 images_folder="images",
                 segmentations_folder="masks"):
        # Paths
        self.images_path = os.path.join(archive_path, images_folder)
        self.segmentations_path = os.path.join(archive_path, segmentations_folder)

        # Caches
        self.images = {}
        self.masks = {}

        self.__extract_segmented_slices__()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice = self.slices[idx]

        if slice.file_name in self.images:
            image = self.images[slice.file_name]
        else:
            image_path = slice.build_path(self.images_path)
            image = np.load(image_path)
            self.images[slice.file_name] = image

        if slice.file_name in self.masks:
            mask = self.masks[slice.file_name]
        else:
            segmentation_path = slice.build_path(self.segmentations_path)
            mask = np.load(segmentation_path)
            self.masks[slice.file_name] = mask

        return image, mask

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
                slices = [f for f in os.listdir(examination_path) if f.endswith(".npy")]

                for slice in slices:
                    self.slices.append(CineMRISlice(slice, patient_id, examination_id))


def main():
    archive_path = "../../data/cinemri_mha/rijnstate"
    subset_path = "../../data/cinemri_mha/segmentation_subset"
    dataset = SegmentationDataset(subset_path)

    for i in range(10):
        image, mask = dataset[i]

        plt.figure()
        plt.imshow(image, cmap="gray")
        masked = np.ma.masked_where(mask == 0, mask)
        plt.imshow(masked, cmap='autumn', alpha=0.2)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()