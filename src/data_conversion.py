#!/usr/local/bin/python3
import sys
import argparse
from pathlib import Path
import numpy as np
import json
import random
import cv2
from skimage import io
import SimpleITK as sitk
from cinemri.utils import get_patients
import utils
from sklearn.model_selection import KFold


# TODO: see if there is loss in quality when converting to .png
def save_frame(source_path, target_path, index=0):
    """
    Saves a single frame of  .mha cine-MRI slice as .npy and .png
    :param source_path: a path to the original file
    :param target_path: a path to save .npy and .png files
    :param index: an index of a frame to extract
    :return:
    """
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
    """
    Extracts a subset of the archive only related to segmentation

    :param archive_path: a path to the full cine-MRI data archive
    :param destination_path: a path to save the extracted segmentation subset
    :param images_folder: a name of the images folder in the archive
    :param segmentations_folder: a name of the folder with cavity segmentations in the archive
    :param target_images_folder: a name of the images folder in the segmentation subset
    :param target_segmentations_folder: a name of the folder with cavity segmentations in the segmentation subset
    :return:
    """

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
        # Skip patients without slices
        if len(patient.slices()) == 0:
            continue

        # Create patient folder in both images and masks target directories
        patient_images_path = target_images_path / patient.id
        patient_images_path.mkdir(exist_ok=True)

        patient_segmentations_path = target_segmentations_path / patient.id
        patient_segmentations_path.mkdir(exist_ok=True)

        for (examination_id, slices) in patient.examinations.items():
            # Skip examinations without slices
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


def extract_segmentation(argv):
    """
        A command line wrapper of extract_segmentation_data

        Command line arguments
        :param argv: command line arguments
        :return:
        """

    parser = argparse.ArgumentParser()
    parser.add_argument('archive_path', type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument('destination_path', type=str, help="a path to save the extracted segmentation subset")
    parser.add_argument('--images', type=str, default="images", help="a name of the images folder in the archive")
    parser.add_argument('--masks', type=str, default="cavity_segmentations", help="a name of the folder with cavity segmentations in the archive")
    parser.add_argument('--target_images', type=str, default="images", help="a name of the images folder in the segmentation subset")
    parser.add_argument('--target_masks', type=str, default="masks", help="a name of the folder with cavity segmentations in the segmentation subset")
    args = parser.parse_args(argv)

    archive_path = Path(args.archive_path)
    destination_path = Path(args.destination_path)
    images_folder = args.images
    segmentations_folder = args.masks
    target_images_folder = args.target_images
    target_segmentations_folder = args.target_masks

    extract_segmentation_data(archive_path,
                              destination_path,
                              images_folder,
                              segmentations_folder,
                              target_images_folder,
                              target_segmentations_folder)


def convert_2d_image_to_pseudo_3d(input_filename, output_filename_stem, file_format = ".nii.gz",
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
    img = np.load(input_filename) if input_filename.suffix == ".npy" else io.imread(input_filename)

    assert len(img.shape) == 2, 'images should be grayscalse'
    img = img[None]  # add dimension

    if is_seg:
        img = img.astype(np.uint32)

    itk_img = sitk.GetImageFromArray(img)
    itk_img.SetSpacing(spacing[::-1])
    sitk.WriteImage(itk_img, str(output_filename_stem) + file_format)


def subset_to_diag_nnunet(patients,
                          segmentation_path,
                          target_path,
                          images_folder="images",
                          masks_folder="masks",
                          is_train=True):
    """
    Saves images an masks for training or test subset of patients
    :param patients: a list of patients in the subset
    :param segmentation_path: a path to the segmentation dataset
    :param target_path: a path to save the subset
    :param images_folder: an images folder name
    :param masks_folder: a masks folder name
    :param is_train: a boolean flag indicating if it is a training subset
    :return:
    """

    # Create folders of the subset
    target_path.mkdir(exist_ok=True)

    train_path_imags = target_path / images_folder
    train_path_imags.mkdir(exist_ok=True)

    train_path_masks = target_path / masks_folder
    train_path_masks.mkdir(exist_ok=True)

    # Extract and save files related to the specified patients list
    for patient in patients:
        for examination_id, slices in patient.examinations.items():
            for slice in slices:
                # Filter out .png images
                if slice.endswith(".npy"):
                    separator = "_"
                    file_format = ".mha" if is_train else ".nii.gz"
                    file_id = separator.join([patient.id, examination_id, slice[:-4]])

                    image_id = file_id if is_train else (file_id + "_0000")
                    image_stem = train_path_imags / image_id
                    slice_image_path = segmentation_path / images_folder / patient.id / examination_id / slice
                    convert_2d_image_to_pseudo_3d(slice_image_path, image_stem, file_format=file_format)

                    mask_stem = train_path_masks / file_id
                    slice_mask_path = segmentation_path / masks_folder / patient.id / examination_id / slice
                    convert_2d_image_to_pseudo_3d(slice_mask_path, mask_stem, file_format=file_format, is_seg=False)


def create_folds(data_path,
                 train_patients_ids,
                 train_folder="train",
                 images_folder="images",
                 folds_num=5,
                 folds_file="splits_final.json"):

    """
    Creates custom folds for nnU-Net stratified by patients
    :param data_path: a path to a training subset
    :param train_patients_ids: a numpy array of patients' ids in the training subset
    :param train_folder: a name of the folder with training data
    :param images_folder: a name of the folder with images
    :param folds_file: a name of a file to save folds split
    :param folds_num: a number of folds
    :return:
    """

    random.shuffle(train_patients_ids)
    kf = KFold(n_splits=folds_num)
    folds = []

    train_images_path = data_path / train_folder / images_folder
    train_image_files = train_images_path.glob("*.mha")
    train_image_ids = [f.stem for f in train_image_files]
    for train_index, val_index in kf.split(train_patients_ids):
        fold_train_ids = train_patients_ids[train_index]
        fold_val_ids = train_patients_ids[val_index]

        # Extract scans ids
        fold_train_scans_ids = [ind for ind in train_image_ids if (ind.split("_")[0] in fold_train_ids)]
        fold_val_scans_ids = [ind for ind in train_image_ids if (ind.split("_")[0] in fold_val_ids)]

        fold = {"train": fold_train_scans_ids, "val": fold_val_scans_ids}
        folds.append(fold)

    destination_file_path = data_path / train_folder / folds_file
    with open(destination_file_path, "w") as f:
        json.dump(folds, f)


def convert_to_diag_nnunet(segmentation_path,
                           target_path,
                           images_folder="images",
                           masks_folder="masks",
                           train_folder="train",
                           test_folder="test",
                           train_split=0.8):
    """
    Converts the segmentation data subset to a format a dockerized version of nnU-Net from DIAG
    can convert to nnU-Net input format

    :param  segmentation_path : path to a segmentation subset of cine-MRI data
    :param  target_path: a destination path to save converted files
    :param  images_folder : a folder inside the archive, which contains scans
    :param  masks_folder : a folder inside the archive, which contains masks
    :param  train_folder : a name of a folder with training data
    :param  test_folder : a name of a folder with test data
    :param  train_proportion : a share of the data to use for training
    :return:
    """

    # Make directories to save converted images
    target_path.mkdir(exist_ok=True)

    train_patients, test_patients = utils.train_test_split(segmentation_path,
                                                           target_path,
                                                           images_folder=masks_folder,
                                                           train_proportion=train_split)

    # Convert training data
    subset_to_diag_nnunet(train_patients,
                          segmentation_path,
                          target_path / train_folder,
                          images_folder,
                          masks_folder)

    # Convert test data
    subset_to_diag_nnunet(test_patients,
                          segmentation_path,
                          target_path / test_folder,
                          images_folder,
                          masks_folder,
                          is_train=False)

    # Split into folds
    train_patient_ids = np.array([patient.id for patient in train_patients])
    create_folds(target_path,
                 train_patient_ids,
                 train_folder=train_folder,
                 images_folder=images_folder)


def to_diag_nnunet(argv):
    """
    A command line wrapper of convert_to_diag_nnunet

    :param argv: command line arguments
    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation_path', type=str, help="path to a segmentation subset of cine-MRI data")
    parser.add_argument('target_path', type=str, help="a destination path to save converted files")
    parser.add_argument('--images', type=str, default="images", help="a folder inside the archive, which contains scans")
    parser.add_argument('--masks', type=str, default="masks", help="a folder inside the archive, which contains masks")
    parser.add_argument('--train', type=str, default="train", help="a name of a folder with training data")
    parser.add_argument('--test', type=str, default="test", help="a name of a folder with test data")
    parser.add_argument('--train_split', type=str, default=0.8, help="a share of the data to use for training")
    args = parser.parse_args(argv)

    segmentation_path = Path(args.segmentation_path)
    target_path = Path(args.target_path)
    images_folder = args.images
    masks_folder = args.masks
    train_folder = args.train
    test_folder = args.test
    train_split = args.train_split

    convert_to_diag_nnunet(segmentation_path,
                           target_path,
                           images_folder,
                           masks_folder,
                           train_folder,
                           test_folder,
                           train_split)


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")
    diag_nnUNet_path = Path("../../data/cinemri_mha/diag_nnunet")

    convert_to_diag_nnunet(subset_path, diag_nnUNet_path)

    """
    unique_shapes = find_unique_shapes(archive_path, "cavity_segmentations")
    print("Unique scan dimensions in the dataset")
    print(unique_shapes)
    """

    # extract_segmentation_data(archive_path, subset_path)


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(99)

    # Very first argument determines action
    actions = {
        'extract_segmentation': extract_segmentation,
        'to_diag_nnunet': to_diag_nnunet
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: nnunet ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
