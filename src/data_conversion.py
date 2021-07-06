#!/usr/local/bin/python3

import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import json
import random
import shutil
from skimage import io
import SimpleITK as sitk
from cinemri.config import ARCHIVE_PATH
from cinemri.utils import get_patients, get_image_orientation
from utils import slices_full_ids_from_patients
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from config import *


def save_frame(source_path, target_path, index=0, is_segm=False):
    """Saves a single frame of  .mha cine-MRI slice as .npy and .png

    Parameters
    ----------
    source_path : Path
       A path to the original file
    target_path : Path
       A path to save .npy and .png files
    index : int
       An index of a frame to extract

    """

    frame = sitk.GetArrayFromImage(sitk.ReadImage(str(source_path)))[index]
    if is_segm:
        frame[frame > 0] = 1

    frame_target_path = target_path / source_path.stem
    # Save .npy
    np.save(str(frame_target_path) + ".npy", frame)
    # Save .png
    frame_png = frame / frame.max() * 255
    frame_png = frame_png.astype(np.uint8)
    io.imsave(str(frame_target_path) + ".png", frame_png)


def find_segmented_frame_index(segmentation_path):

    image = sitk.GetArrayFromImage(sitk.ReadImage(str(segmentation_path)))

    for index, frame in enumerate(image):
        # Find first frame with non zero values
        if len(np.nonzero(frame)[0]) > 0:
            return index

    # If not found return None
    return None


def extract_segmentation_data(archive_path,
                              destination_path,
                              images_folder="images",
                              segmentations_folder="cavity_segmentations",
                              target_images_folder="images",
                              target_segmentations_folder="masks"):
    """Extracts a subset of the archive only related to segmentation

    Parameters
    ----------
    archive_path : Path
       A path to the full cine-MRI data archive
    destination_path : Path
       A path to save the extracted segmentation subset
    images_folder : str, default="images"
       A name of the images folder in the archive
    segmentations_folder : str, default="cavity_segmentations"
       A name of the folder with cavity segmentations in the archive
    target_images_folder : str, default="images"
       A name of the images folder in the segmentation subset
    target_segmentations_folder : str, default="masks"
       A name of the folder with cavity segmentations in the segmentation subset
    """

    # create target paths and folders
    destination_path = Path(destination_path)
    destination_path.mkdir(exist_ok=True)

    target_images_path = destination_path / target_images_folder
    target_images_path.mkdir(exist_ok=True)

    target_segmentations_path = destination_path / target_segmentations_folder
    target_segmentations_path.mkdir(exist_ok=True)

    images_path = archive_path / images_folder
    segmentation_path = archive_path / segmentations_folder
    patients = get_patients(segmentation_path)
    # Now get all slices for each patient and create an array of Patients
    for patient in patients:
        # Create patient folder in both images and masks target directories
        patient.build_path(target_images_path).mkdir(exist_ok=True)
        patient.build_path(target_segmentations_path).mkdir(exist_ok=True)

        for study in patient.studies:
            # Skip studies without slices
            if len(study.slices) == 0:
                continue

            # Create study folder in both images and masks target directories
            study_images_path = study.build_path(target_images_path)
            study_images_path.mkdir(exist_ok=True)

            study_segmentations_path = study.build_path(target_segmentations_path)
            study_segmentations_path.mkdir(exist_ok=True)

            for slice in study.slices:
                # read a segmentation mask, find the segmented frame index
                mask_path = slice.build_path(segmentation_path)
                segmented_frame_index = find_segmented_frame_index(mask_path)
                if segmented_frame_index is not None:
                    save_frame(mask_path, study_segmentations_path, segmented_frame_index, True)

                    # read an image and extract the segmented frame
                    slice_path = slice.build_path(images_path)
                    save_frame(slice_path, study_images_path, segmented_frame_index)




def extract_segmentation(argv):
    """A command line wrapper of extract_segmentation_data

    Parameters
    ----------
    argv : list of str
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


def create_folds(data_path,
                 train_patients_ids,
                 train_folder="train",
                 images_folder="images",
                 folds_num=5,
                 folds_file="splits_final.json"):
    """Creates custom folds for nnU-Net stratified by patients

    Parameters
    ----------
    data_path : Path
       A path to a training subset
    train_patients_ids : ndarray of str
       An array of patients' ids in the training subset
    train_folder : str, default="train"
       A name of the folder with training data
    images_folder : str, default="images"
       A name of the folder with images
    folds_num : int, default=5
       A number of folds to make
    folds_file : str, default="splits_final.json"
       A name of a file to save folds split
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

        # Extract F ids
        fold_train_scans_ids = [ind for ind in train_image_ids if (ind.split("_")[0] in fold_train_ids)]
        fold_val_scans_ids = [ind for ind in train_image_ids if (ind.split("_")[0] in fold_val_ids)]

        fold = {"train": fold_train_scans_ids, "val": fold_val_scans_ids}
        folds.append(fold)

    destination_file_path = data_path / train_folder / folds_file
    with open(destination_file_path, "w") as f:
        json.dump(folds, f)


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


def extract_frames(slice_path,
                   slice_id,
                   target_path_images,
                   target_path_metadata):
    """
    Extracts frame of a cine-MRI slice, converts each frame to a pseudo 3D image to meet nn-Unet input requirements
    and saves the extracted frames and slice metadata to the specified locations
    Parameters
    ----------
    slice_path : Path
       A path to a cine-MRI slice in format "patientID_studyID_sliceID"
    slice_id : str
       A full id of a slice
    target_path_images : Path
       A path where to save the extracted frames
    target_path_metadata : Path
       A path where to save the slice metadata
    """

    img = sitk.ReadImage(str(slice_path))
    depth = img.GetDepth()
    # Check that a slice is valid
    if depth >= 30 and get_image_orientation(img) == "ASL":
        metadata = {
                    "Spacing": img.GetSpacing(),
                    "Origin": img.GetOrigin(),
                    "Direction": img.GetDirection(),
                    "PatientID": img.GetMetaData("PatientID"),
                    "StudyInstanceUID": img.GetMetaData("StudyInstanceUID"),
                    "SeriesInstanceUID": img.GetMetaData("SeriesInstanceUID")
                    }

        if img.HasMetaDataKey("Sex"):
            metadata["Sex"] = img.GetMetaData("Sex")

        if img.HasMetaDataKey("Age"):
            metadata["Age"] = img.GetMetaData("Age")

        metadata_file_path = target_path_metadata / (slice_id + ".json")
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f)

        img_array = sitk.GetArrayFromImage(img)
        for ind, frame in enumerate(img_array):
            frame_2d = convert_2d_image_to_pseudo_3d(frame)
            # 0000 suffix is necessary for nn-UNet
            niigz_path = target_path_images / (slice_id + "_" + str(ind) + "_0000.nii.gz")
            sitk.WriteImage(frame_2d, str(niigz_path))
    else:
        print("Skipping incomplete series or series with different anatomical plane, slice id: {}".format(slice_id))


# Scans have adidtional "_0000" suffix, masks do not have it
def merge_frames(slice_full_id,
                 frames_folder,
                 target_folder,
                 metadata_path,
                 masks=True):
    """
    Merges frames extracted from a cine-MRI slice into the full slice or predicted segmentation masks into
    a single file

    Parameters
    ----------
    slice_full_id : str
       A full id of a cine-MRI slice in format "patientID_studyID_sliceID"
    frames_folder : Path
       A path to a folder containing frames of a cine-MRI slice
    target_folder : Path
       A path to a folder where to save a merged cine-MRI slice
    metadata_path : Path
       A path to a folder containing metadata of a cine-MRI slice
    masks : bool, default=True
       A flag indicating whether frames are from a cine-MRI slice or predicted masks
    """

    frame_files_glob = frames_folder.glob(slice_full_id + "*.nii.gz")
    # Sort by file index to merge the images in the correct order
    sort_index = -1 if masks else -2
    files = sorted([file for file in frame_files_glob], key=lambda file: int(file.name[:-7].split("_")[sort_index]))

    image = []
    for frame_file in files:
        frame = sitk.ReadImage(str(frame_file))
        image.append(sitk.GetArrayFromImage(frame)[0])

    image = np.array(image).astype(np.uint8)
    sitk_image = sitk.GetImageFromArray(image)

    # Extract and assign metadata
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    sitk_image.SetOrigin(tuple(metadata["Origin"]))
    sitk_image.SetSpacing(tuple(metadata["Spacing"]))
    sitk_image.SetDirection(tuple(metadata["Direction"]))
    sitk_image.SetMetaData("PatientID", metadata["PatientID"])
    sitk_image.SetMetaData("StudyInstanceUID", metadata["StudyInstanceUID"])
    sitk_image.SetMetaData("SeriesInstanceUID", metadata["SeriesInstanceUID"])

    try:
        sitk_image.SetMetaData("Sex", metadata["Sex"])
        sitk_image.SetMetaData("Age", metadata["Age"])
    except:
        pass

    # Save image
    slice_id = slice_full_id.split(SEPARATOR)
    image_path = target_folder / (slice_id[-1] + ".mha")
    sitk.WriteImage(sitk_image, str(image_path))


def save_visualised_prediction(images_path, predictions_path, target_path, save_gif=True):
    """
    Visualises the predicted masks as overlays on the frames and saves as a png file
    Parameters
    ----------
    images_path : Path
       A path to a folder that contains frames extracted from a cine-MRI slice
    predictions_path : Path
       A path to a folder that contains predicted segmentation masks
    target_path : Path
       A path to a folder to save visualized prediction
    save_gif : bool, default=True
       A boolean flag indicating whether visualized prediction should be merged into a gif and saved
    """

    target_path.mkdir(exist_ok=True)

    png_path = target_path / "pngs"
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
        plt.savefig(overlayed_mask_file_path, bbox_inches='tight', pad_inches=0)
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
            str(target_path) + "/" + frame_id[:-3] + ".gif",
        ]
        subprocess.run(command)


def save_visualised_full_prediction(images_path, predictions_path, target_path, save_gif=True):
    """
    Visualises the predicted masks as overlays on the frames and saves as a png file
    Parameters
    ----------
    images_path : Path
       A path to a folder that contains cine-MRI slices
    predictions_path : Path
       A path to a folder that contains predicted segmentation masks
    target_path : Path
       A path to a folder to save visualized prediction
    save_gif : bool, default=True
       A boolean flag indicating whether visualized prediction should be merged into a gif and saved
    """

    target_path.mkdir(exist_ok=True)

    # get frames ids
    patients = get_patients(images_path)
    for patient in patients:
        patient.build_path(target_path).mkdir()

        for study in patient.studies:
            study.build_path(target_path).mkdir()

            for slice in study.slices:
                # extract cine-MRI slice
                slice_path = slice.build_path(images_path)
                slice_img = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
                # extract predicted mask
                mask_path = slice.build_path(predictions_path)
                mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

                vis_path = slice.build_path(target_path, extension="")
                vis_path.mkdir()

                # Make a separate folder for .png files
                png_path = vis_path / "pngs"
                png_path.mkdir()

                for ind, frame in enumerate(slice_img):
                    frame_mask = mask[ind]

                    plt.figure()
                    plt.imshow(frame, cmap="gray")
                    masked = np.ma.masked_where(frame_mask == 0, frame_mask)
                    plt.imshow(masked, cmap='autumn', alpha=0.2)
                    plt.axis("off")
                    overlayed_mask_file_path = png_path / "{}_{}.png".format(slice.id, ind)
                    plt.savefig(overlayed_mask_file_path, bbox_inches='tight', pad_inches=0)
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
                        str(vis_path) + "/" + slice.id + ".gif",
                    ]
                    subprocess.run(command)


def extract_detection_gifs(source_path, target_path):

    target_path.mkdir(exist_ok=True)

    patient_ids = [f.name for f in source_path.iterdir() if f.is_dir()]
    for patient_id in patient_ids:
        patient_path = source_path / patient_id
        studies = [f.name for f in patient_path.iterdir() if f.is_dir()]

        for study_id in studies:
            study_path = patient_path / study_id
            slices = [f.name for f in study_path.iterdir() if f.is_dir()]

            for slice_id in slices:
                gif_source_path = study_path / slice_id / (slice_id + ".gif")
                gif_target_path = target_path / (SEPARATOR.join((patient_id, study_id, slice_id)) + ".gif")

                shutil.copy(gif_source_path, gif_target_path)



def test():
    archive_path = ARCHIVE_PATH
    subset_path = Path("segmentation_subset")
    diag_nnUNet_path = Path("../../data/cinemri_mha/diag_nnunet")

    detection_path = Path(DETECTION_PATH)
    # Folders
    destination_path = Path(DETECTION_PATH) / SEGM_FRAMES_FOLDER
    images_folder = IMAGES_FOLDER
    segmentation_folder = SEGMENTATION_FOLDER
    diag_nnunet_folder = Path(DETECTION_PATH) / DIAG_NNUNET_FOLDER

    extract_segmentation_data(archive_path, destination_path, images_folder, segmentation_folder)
    
    convert_to_diag_nnunet(destination_path, diag_nnunet_folder)

    """
    images_path = Path("../../data/cinemri_mha/detection_new/images")
    prediction_path = Path("../../data/cinemri_mha/detection_new/full_segmentation/merged_masks")
    target_path = Path("../../experiments/detection_vis_new")
    target_path1 = Path("../../experiments/detection_gifs_new")

    save_visualised_full_prediction(images_path, prediction_path, target_path)
    extract_detection_gifs(target_path, target_path1)
    """

    #extract_frames(subset_path, diag_nnUNet_path)

    """
    unique_shapes = find_unique_shapes(archive_path, "cavity_segmentations")
    print("Unique scan dimensions in the dataset")
    print(unique_shapes)
    """

    #extract_segmentation_data(archive_path, subset_path)

    # extract_frames(Path("1.3.12.2.1107.5.2.30.26380.2019031314281933334670409.0.0.0.mha"), Path("frames"))
    # merge_frames(Path("../full_pred_test/prediction"), Path("merged_prediction"), Path("frames/metadata.json"))
    #save_visualised_prediction(Path("frames"), Path("prediction"), Path("vis"))


if __name__ == '__main__':
    test()

    """
    np.random.seed(99)
    random.seed(99)

    # Very first argument determines action
    actions = {
        "extract_segmentation": extract_segmentation,
        "to_diag_nnunet": to_diag_nnunet
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: data_conversion ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
    """
