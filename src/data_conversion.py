#!/usr/local/bin/python3

import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import json
import random
from skimage import io
import SimpleITK as sitk
from cinemri.utils import get_patients
import utils
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


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
    image_png = image / image.max() * 255
    image_png = image_png.astype(np.uint8)
    io.imsave(str(image_target_path.with_suffix(".png")), image_png)


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

    patients = get_patients(archive_path / segmentations_folder)
    # Now get all scans for each patient and create an array of Patients
    for patient in patients:
        # Create patient folder in both images and masks target directories
        patient_images_path = target_images_path / patient.id
        patient_images_path.mkdir(exist_ok=True)

        patient_segmentations_path = target_segmentations_path / patient.id
        patient_segmentations_path.mkdir(exist_ok=True)

        for (scan_id, slices) in patient.scans.items():
            # Skip scans without slices
            if len(slices) == 0:
                continue

            # Create scan folder in both images and masks target directories
            scan_images_path = patient_images_path / scan_id
            scan_images_path.mkdir(exist_ok=True)

            scan_segmentations_path = patient_segmentations_path / scan_id
            scan_segmentations_path.mkdir(exist_ok=True)

            for slice in slices:
                # read an image and extract the first frame
                slice_path = Path(archive_path) / images_folder / patient.id / scan_id / slice
                save_frame(slice_path, scan_images_path)

                # read a segmentation mask and extract the first frame
                segmentation_path = Path(archive_path) / segmentations_folder / patient.id / scan_id / slice
                save_frame(segmentation_path, scan_segmentations_path)


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


def convert_2d_image_file_to_pseudo_3d(input_file_path, spacing=[999, 1, 1], is_seg= False):
    """
    Reads an image (must be .npy or fromat recognized by skimage) and converts it into a series of niftis.
    The image should be grayscalse
    :param input_file_path: a path to image to convert
    :param spacing:
    :param is_seg: is the specified image a segmentation mask
    :return: an image converted to pseudo 2d format suitable for nnU-Net
    """
    img = np.load(input_file_path) if input_file_path.suffix == ".npy" else io.imread(input_file_path)
    return convert_2d_image_to_pseudo_3d(img, spacing, is_seg)


def convert_2d_image_to_pseudo_3d(image, spacing=[999, 1, 1], is_seg=False):
    """
    Taken from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/file_conversions.py
    and slightly modified
    Converts an image into a series of niftis.
    The image should be grayscalse
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    Segmentations will be converted to np.uint32!
    :param image: an image to convert
    :param spacing:
    :param is_seg: is the specified image a segmentation mask
    :return: an image converted to pseudo 2d format suitable for nnU-Net
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
        for scan_id, slices in patient.scans.items():
            for slice in slices:
                # Filter out .png images
                if slice.endswith(".npy"):
                    separator = "_"
                    file_format = ".mha" if is_train else ".nii.gz"
                    file_id = separator.join([patient.id, scan_id, slice[:-4]])

                    image_id = file_id if is_train else (file_id + "_0000")
                    image_stem = train_path_imags / image_id
                    slice_image_path = segmentation_path / images_folder / patient.id / scan_id / slice
                    img_pseudo_3d = convert_2d_image_file_to_pseudo_3d(slice_image_path, image_stem, file_format=file_format)
                    sitk.WriteImage(img_pseudo_3d, str(image_stem) + file_format)

                    mask_stem = train_path_masks / file_id
                    slice_mask_path = segmentation_path / masks_folder / patient.id / scan_id / slice
                    mask_pseudo_3d = convert_2d_image_file_to_pseudo_3d(slice_mask_path, mask_stem, file_format=file_format, is_seg=False)
                    sitk.WriteImage(mask_pseudo_3d, str(mask_stem) + file_format)


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
    :param  train_split : a share of the data to use for training
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


def mha_to_niigz(mha_path,
                 target_folder):
    img = sitk.ReadImage(str(mha_path))
    niigz_path = target_folder / (mha_path.stem + "_0000.nii.gz")
    sitk.WriteImage(img, str(niigz_path))


def extract_frames(mha_image_path,
                   target_folder):

    target_folder.mkdir(exist_ok=True)
    img = sitk.ReadImage(str(mha_image_path))
    metadata = {
                "Spacing": img.GetSpacing(),
                "Origin": img.GetOrigin(),
                "Direction": img.GetDirection(),
                "PatientID": img.GetMetaData("PatientID"),
                "StudyInstanceUID": img.GetMetaData("StudyInstanceUID"),
                "SeriesInstanceUID": img.GetMetaData("SeriesInstanceUID")
                }

    metadata_file_path = target_folder / "metadata.json"
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f)

    img_array = sitk.GetArrayFromImage(img)
    for ind, frame in enumerate(img_array):
        frame_2d = convert_2d_image_to_pseudo_3d(frame)
        niigz_path = target_folder / (mha_image_path.stem + "_" + str(ind) + "_0000.nii.gz")
        sitk.WriteImage(frame_2d, str(niigz_path))


# Scans have adidtional "_0000" suffix, masks do not have it
def merge_frames(frames_folder,
                 target_folder,
                 metadata_path,
                 masks=True):

    target_folder.mkdir(exist_ok=True)
    frame_files_glob = frames_folder.glob("*.nii.gz")
    image = []
    # Sort by file index
    sort_index = -1 if masks else -2
    files = sorted([file for file in frame_files_glob], key=lambda file: int(file.name[:-7].split("_")[sort_index]))
    for frame_file in files:
        frame = sitk.ReadImage(str(frame_file))
        image.append(sitk.GetArrayFromImage(frame)[0])

    image = np.array(image)
    stem_shift = -10 if masks else -15
    image_path = target_folder / (frame_file.name[:stem_shift] + ".mha")
    sitk_image = sitk.GetImageFromArray(image)
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    sitk_image.SetOrigin(tuple(metadata["Origin"]))
    sitk_image.SetSpacing(tuple(metadata["Spacing"]))
    sitk_image.SetDirection(tuple(metadata["Direction"]))
    sitk_image.SetMetaData("PatientID", metadata["PatientID"])
    sitk_image.SetMetaData("StudyInstanceUID", metadata["StudyInstanceUID"])
    sitk_image.SetMetaData("SeriesInstanceUID", metadata["SeriesInstanceUID"])
    sitk.WriteImage(sitk_image, str(image_path))


# Visualize the predicted mask as overlay on the frame and saved as png file
def save_visualised_prediction(images_path, predictions_path, png_path, save_gif=True):

    png_path.mkdir(exist_ok=True)

    # get frames ids
    files = images_path.glob("*.nii.gz")
    frame_ids = sorted([file.name[:-12] for file in files], key=lambda file_id: int(file_id.split("_")[-1]))

    for frame_id in frame_ids:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(images_path / (frame_id + "_0000.nii.gz"))))[0]
        nnUNet_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(predictions_path / (frame_id + ".nii.gz"))))[0]

        plt.figure()
        plt.imshow(image, cmap="gray")
        masked = np.ma.masked_where(nnUNet_mask == 0, nnUNet_mask)
        plt.imshow(masked, cmap='autumn', alpha=0.2)
        plt.axis("off")
        overlayed_mask_file_path = png_path / (frame_id + ".png")
        plt.savefig(overlayed_mask_file_path)
        plt.close()

    if save_gif:
        command = [
            "convert",
            "-coalesce",
            "-delay",
            "20",
            "-loop",
            "0",
            str(png_path) + "/*png",
            str(png_path) + "/" + frame_id[:-3] + ".gif",
        ]
        subprocess.run(command)


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    subset_path = Path("../../data/cinemri_mha/segmentation_subset")
    diag_nnUNet_path = Path("../../data/cinemri_mha/diag_nnunet")

    extract_frames(subset_path, diag_nnUNet_path)

    """
    unique_shapes = find_unique_shapes(archive_path, "cavity_segmentations")
    print("Unique scan dimensions in the dataset")
    print(unique_shapes)
    """

    # extract_segmentation_data(archive_path, subset_path)

    # extract_frames(Path("1.3.12.2.1107.5.2.30.26380.2019031314281933334670409.0.0.0.mha"), Path("frames"))
    # merge_frames(Path("../full_pred_test/prediction"), Path("merged_prediction"), Path("frames/metadata.json"))
    save_visualised_prediction(Path("frames"), Path("prediction"), Path("vis"))


if __name__ == '__main__':
    """
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
        print('Usage: data_conversion ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
    """

