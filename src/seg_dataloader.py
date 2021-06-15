from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from cinemri.definitions import CineMRISlice
import SimpleITK as sitk
import matplotlib.pyplot as plt


class SegmentationDatasetFull(Dataset):
    def __init__(self,
                 archive_path,
                 images_folder="images",
                 segmentations_folder="cavity_segmentations"):
        # Paths
        self.images_path = Path(archive_path) / images_folder
        self.segmentations_path = Path(archive_path) / segmentations_folder

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
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))[0]
            self.images[slice.file_name] = image

        if slice.file_name in self.masks:
            mask = self.masks[slice.file_name]
        else:
            segmentation_path = slice.build_path(self.segmentations_path)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(segmentation_path)))[0]
            self.masks[slice.file_name] = mask

        return image, mask

    def __extract_segmented_slices__(self):
        # Patient ids
        self.patient_ids = [f.name for f in self.segmentations_path.iterdir() if f.is_dir()]

        # Now get all studies for each patient and create an array of Patients
        self.slices = []
        for patient_id in self.patient_ids:
            patient_segmentations_path = self.segmentations_path / patient_id
            studies = [f.name for f in patient_segmentations_path.iterdir() if f.is_dir()]

            for study_id in studies:
                study_path = patient_segmentations_path / study_id
                slices = [f.name for f in study_path.iterdir() if f.is_file()]

                for slice in slices:
                    self.slices.append(CineMRISlice(slice.stem, patient_id, study_id))


class SegmentationDataset(Dataset):
    def __init__(self,
                 archive_path,
                 images_folder="images",
                 segmentations_folder="masks"):
        # Paths
        self.images_path = Path(archive_path) / images_folder
        self.segmentations_path = Path(archive_path) / segmentations_folder

        # Caches
        self.images = {}
        self.masks = {}

        self.__extract_segmented_slices__()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice = self.slices[idx]

        if slice.full_id in self.images:
            image = self.images[slice.full_id]
        else:
            image_path = slice.build_path(self.images_path, extension=".npy")
            image = np.load(image_path)
            self.images[slice.full_id] = image

        if slice.full_id in self.masks:
            mask = self.masks[slice.full_id]
        else:
            segmentation_path = slice.build_path(self.segmentations_path, extension=".npy")
            mask = np.load(segmentation_path)
            self.masks[slice.full_id] = mask

        return image, mask

    def __extract_segmented_slices__(self):
        # Patient ids
        self.patient_ids = [f.name for f in self.segmentations_path.iterdir() if f.is_dir()]

        # Now get all studies for each patient and create an array of Patients
        self.slices = []
        for patient_id in self.patient_ids:
            patient_segmentations_path = self.segmentations_path / patient_id
            studies = [f.name for f in patient_segmentations_path.iterdir() if f.is_dir()]

            for study_id in studies:
                study_path = patient_segmentations_path / study_id
                slice_ids = [f.stem for f in study_path.iterdir() if f.suffix == ".npy"]

                for slice_id in slice_ids:
                    self.slices.append(CineMRISlice(slice_id, patient_id, study_id))


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")
    dataset = SegmentationDataset(subset_path)
    #dataset = SegmentationDatasetFull(archive_path)

    for i in range(10):
        image, mask = dataset[i]

        plt.figure()
        plt.imshow(image, cmap="gray")
        masked = np.ma.masked_where(mask == 0, mask)
        plt.imshow(masked, cmap='autumn', alpha=0.2)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    test()