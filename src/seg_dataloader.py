from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from cinemri.utils import CineMRISlice
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

        # Now get all scans for each patient and create an array of Patients
        self.slices = []
        for patient_id in self.patient_ids:
            patient_segmentations_path = self.segmentations_path / patient_id
            scans = [f.name for f in patient_segmentations_path.iterdir() if f.is_dir()]

            for scan_id in scans:
                scan_path = patient_segmentations_path / scan_id
                slices = [f.name for f in scan_path.iterdir() if f.is_file()]

                for slice in slices:
                    self.slices.append(CineMRISlice(patient_id, scan_id, slice.stem))


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
        self.patient_ids = [f.name for f in self.segmentations_path.iterdir() if f.is_dir()]

        # Now get all scans for each patient and create an array of Patients
        self.slices = []
        for patient_id in self.patient_ids:
            patient_segmentations_path = self.segmentations_path / patient_id
            scans = [f.name for f in patient_segmentations_path.iterdir() if f.is_dir()]

            for scan_id in scans:
                scan_path = patient_segmentations_path / scan_id
                slices = [f.name for f in scan_path.iterdir() if f.suffix == ".npy"]

                for slice in slices:
                    self.slices.append(CineMRISlice(patient_id, scan_id, slice.stem))


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