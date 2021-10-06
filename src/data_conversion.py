#!/usr/local/bin/python3
import sys
import random
import argparse
from pathlib import Path
import numpy as np
from skimage import io
import SimpleITK as sitk
from cinemri.utils import get_patients


def convert_2d_image_file_to_pseudo_3d(input_file_path, spacing=[999, 1, 1], is_seg=False):
    """Reads an image (must be .npy or fromat recognized by skimage) and converts it into a series of niftis.

    The input image should be grayscalse.

    Parameters
    ----------
    input_file_path : Path
       A path to image to convert
    spacing : list, default=[999, 1, 1]
    is_seg : bool, default=False
       Indicates if the specified image is a segmentation mask

    Returns
    -------
    SimpleITK Image
       An image converted to pseudo 2d format suitable for nnU-Net

    """
    img = np.load(input_file_path) if input_file_path.suffix == ".npy" else io.imread(input_file_path)
    return convert_2d_image_to_pseudo_3d(img, spacing, is_seg)


def convert_2d_image_to_pseudo_3d(image, spacing=[999, 1, 1], is_seg=False):
    """
    Taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/file_conversions.py and slightly modified

    Converts an image into a series of niftis.
    The image should be grayscalse
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    Segmentations will be converted to np.uint32!

    Parameters
    ----------
    image : ndarray
       An image to convert
    spacing : list, default=[999, 1, 1]
    is_seg : bool, default=False
       Indicates if the specified image is a segmentation mask

    Returns
    -------
    SimpleITK Image
       An image converted to pseudo 2d format suitable for nnU-Net
    """

    assert len(image.shape) == 2, 'images should be grayscalse'
    image = image[None]  # add dimension

    if is_seg:
        image = image.astype(np.uint32)

    itk_image = sitk.GetImageFromArray(image)
    itk_image.SetSpacing(spacing[::-1])
    return itk_image


def subset_to_diag_nnunet(patients,
                          segmentation_path,
                          target_path,
                          images_folder="images",
                          masks_folder="masks",
                          is_train=True):
    """Saves images an masks for training or test subset of patients

    Parameters
    ----------
    patients : list of Patients
       A list of patients in the subset
    segmentation_path : Path
       A path to the segmentation dataset
    target_path : Path
       A path to save the subset
    images_folder : str, default="images"
       An images folder name
    masks_folder :  str, default="masks"
       A masks folder name
    is_train : bool, default=True
       A boolean flag indicating if it is a training subset
    """

    # Create folders of the subset
    target_path.mkdir(exist_ok=True)

    train_path_images = target_path / images_folder
    train_path_images.mkdir(exist_ok=True)

    train_path_masks = target_path / masks_folder
    train_path_masks.mkdir(exist_ok=True)

    # Extract and save files related to the specified patients list
    for patient in patients:
        for slice in patient.cinemri_slices:
            extension = ".mha" if is_train else ".nii.gz"

            image_id = slice.full_id if is_train else (slice.full_id + "_0000")
            image_stem = train_path_images / image_id
            slice_image_path = slice.build_path(segmentation_path / images_folder, extension=".npy")
            img_pseudo_3d = convert_2d_image_file_to_pseudo_3d(slice_image_path)
            sitk.WriteImage(img_pseudo_3d, str(image_stem) + extension)

            mask_stem = train_path_masks / slice.full_id
            slice_mask_path = slice.build_path(segmentation_path / masks_folder, extension=".npy")
            mask_pseudo_3d = convert_2d_image_file_to_pseudo_3d(slice_mask_path, is_seg=True)
            sitk.WriteImage(mask_pseudo_3d, str(mask_stem) + extension)


def convert_to_diag_nnunet(segmentation_path,
                           target_path,
                           train_folder="train",
                           images_folder="images",
                           masks_folder="masks"):
    """Converts the segmentation data subset to a diag nnU-Net input format

    This format is expected by prepare method of a diag nnU-Net that
    converts it to the nnU-Net input format

    Parameters
    ----------
    segmentation_path : Path
       A path to a segmentation subset of cine-MRI data
    target_path : Path
       A destination path to save converted files
    images_folder : str, default="images"
       A name of a folder that contains scans inside the archive
    masks_folder : str, default="masks"
       A name of a folder that contains masks inside the archive
    train_folder : str, default="train"
       A name of a folder with training data
    """

    # Make directories to save converted images
    target_path.mkdir(exist_ok=True)

    patients = get_patients(segmentation_path / images_folder, slice_extension=".npy")

    # Convert training data
    subset_to_diag_nnunet(patients,
                          segmentation_path,
                          target_path / train_folder,
                          images_folder,
                          masks_folder)


def to_diag_nnunet(argv):
    """A command line wrapper of convert_to_diag_nnunet

    Parameters
    ----------
    argv : list of str
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation_path', type=str, help="path to a segmentation subset of cine-MRI data")
    parser.add_argument('target_path', type=str, help="a destination path to save converted files")
    parser.add_argument('--images', type=str, default="images", help="a folder inside the archive, which contains scans")
    parser.add_argument('--masks', type=str, default="masks", help="a folder inside the archive, which contains masks")
    parser.add_argument('--train', type=str, default="train", help="a name of a folder with training data")
    args = parser.parse_args(argv)

    segmentation_path = Path(args.segmentation_path)
    target_path = Path(args.target_path)
    images_folder = args.images
    masks_folder = args.masks
    train_folder = args.train

    convert_to_diag_nnunet(segmentation_path,
                           target_path,
                           train_folder,
                           images_folder,
                           masks_folder)


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(99)

    # Very first argument determines action
    actions = {
        "to_diag_nnunet": to_diag_nnunet
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: data_conversion ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
