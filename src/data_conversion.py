from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk
from cinemri.utils import get_patients

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


def convert_2d_image_to_pseudo_3d(input_filename: str, output_filename_stem: str, file_format = ".nii.gz",
                                  spacing=[999, 1, 1], is_seg: bool = False) -> None:
    """
    Taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/file_conversions.py
    and slightly modified
    Reads an image (must be a .npy format) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param input_filename:
    :param output_filename_stem: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    img = np.load(input_filename)

    assert len(img.shape) == 2, 'images should grayscalse'
    img = img[None]  # add dimension

    if is_seg:
        img = img.astype(np.uint32)

    itk_img = sitk.GetImageFromArray(img)
    itk_img.SetSpacing(spacing[::-1])
    sitk.WriteImage(itk_img, str(output_filename_stem) + file_format)


def convert_to_pre_nnUnet(segmentation_path,
                          target_path,
                          images_folder="images",
                          masks_folder="masks"):
    """
    Converts the segmentation data subset to a format a dockerized version of nnU-Net from DIAG
    can convert to nnU-Net input fromat

    Parameters
    ----------
        segmentation_path : path to a segmentation subset of cine-MRI data
        target_path: a destination path to save converted files
        images_folder : a folder inside the archive, which contains scans
        masks_folder : a folder inside the archive, which contains masks
    """

    # Make directories to save converted images
    target_path.mkdir(exist_ok=True)

    target_path_imags = target_path / images_folder
    target_path_imags.mkdir(exist_ok=True)

    target_path_masks = target_path / masks_folder
    target_path_masks.mkdir(exist_ok=True)

    patients = get_patients(subset_path)

    for patient in patients:
        for examination_id, slices in patient.examinations.items():
            for slice in slices:
                # Filter out .png images
                if slice.endswith(".npy"):
                    separator = "_"
                    file_id = separator.join([patient.id, examination_id, slice[:-4]])
                    image_stem = target_path_imags / file_id
                    slice_image_path = segmentation_path / images_folder / patient.id / examination_id / slice
                    convert_2d_image_to_pseudo_3d(slice_image_path, image_stem, file_format=".mha")

                    mask_stem = target_path_masks / file_id
                    slice_mask_path = segmentation_path / masks_folder / patient.id / examination_id / slice
                    convert_2d_image_to_pseudo_3d(slice_mask_path, mask_stem, file_format=".mha", is_seg=False)


# TODO: probably save conversion between ids if we use this
def convert_to_pre_nnUnet_custom_ids(segmentation_path,
                                     target_path,
                                     images_folder="images",
                                     masks_folder="masks"):
    """
    Converts the segmentation data subset to a format a dockerized version of nnU-Net from DIAG
    can convert to nnU-Net input fromat

    Parameters
    ----------
        segmentation_path : path to a segmentation subset of cine-MRI data
        target_path: a destination path to save converted files
        images_folder : a folder inside the archive, which contains scans
        masks_folder : a folder inside the archive, which contains masks
    """

    # Make directories to save converted images
    target_path.mkdir(exist_ok=True)

    target_path_imags = target_path / images_folder
    target_path_imags.mkdir(exist_ok=True)

    target_path_masks = target_path / masks_folder
    target_path_masks.mkdir(exist_ok=True)

    patients = get_patients(segmentation_path)

    for patient in patients:
        for examination_ind, (examination_id, slices) in enumerate(patient.examinations.items()):
            slice_ind = 1
            for slice in slices:
                # Filter out .png images
                if slice.endswith(".npy"):
                    separator = "_"
                    file_id = separator.join([patient.id, "scan"+str(examination_ind+1), "slice"+str(slice_ind)])
                    image_stem = target_path_imags / file_id
                    slice_image_path = segmentation_path / images_folder / patient.id / examination_id / slice
                    convert_2d_image_to_pseudo_3d(slice_image_path, image_stem, file_format=".mha")

                    mask_stem = target_path_masks / file_id
                    slice_mask_path = segmentation_path / masks_folder / patient.id / examination_id / slice
                    convert_2d_image_to_pseudo_3d(slice_mask_path, mask_stem, file_format=".mha", is_seg=False)

                    slice_ind += 1


if __name__ == '__main__':
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")
    pre_nnUNet_path = Path("../../data/cinemri_mha/pre_nnUNet_custom")

    #convert_to_pre_nnUnet_custom_ids(subset_path, pre_nnUNet_path)

    """
    unique_shapes = find_unique_shapes(archive_path, "cavity_segmentations")
    print("Unique scan dimensions in the dataset")
    print(unique_shapes)
    """

    #extract_segmentation_data(archive_path, subset_path)
