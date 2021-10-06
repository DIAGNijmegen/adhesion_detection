# Functions related to extraction of different type of data
import sys
import shutil
import argparse
import random
import numpy as np
import json
from pathlib import Path
from skimage import io
import SimpleITK as sitk
from config import SEPARATOR
from utils import slice_complete_and_sagittal, slices_from_full_ids_file
from data_conversion import convert_2d_image_to_pseudo_3d


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


def extract_frames(slice,
                   slice_id,
                   target_path_images,
                   target_path_metadata):
    """
    Extracts frame of a cine-MRI slice, converts each frame to a pseudo 3D image to meet nn-Unet input requirements
    and saves the extracted frames and slice metadata to the specified locations
    Parameters
    ----------
    slice : SimpleITK.Image
       A a cine-MRI slice image
    slice_id : str
       A full id of a slice
    target_path_images : Path
       A path where to save the extracted frames
    target_path_metadata : Path
       A path where to save the slice metadata
    """

    # Check that a slice is valid
    if slice_complete_and_sagittal(slice):
        metadata = {
                    "Spacing": slice.GetSpacing(),
                    "Origin": slice.GetOrigin(),
                    "Direction": slice.GetDirection(),
                    "PatientID": slice.GetMetaData("PatientID"),
                    "StudyInstanceUID": slice.GetMetaData("StudyInstanceUID"),
                    "SeriesInstanceUID": slice.GetMetaData("SeriesInstanceUID")
                    }

        if slice.HasMetaDataKey("Sex"):
            metadata["Sex"] = slice.GetMetaData("Sex")

        if slice.HasMetaDataKey("Age"):
            metadata["Age"] = slice.GetMetaData("Age")

        metadata_file_path = target_path_metadata / (slice_id + ".json")
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f)

        img_array = sitk.GetArrayFromImage(slice)
        for ind, frame in enumerate(img_array):
            frame_2d = convert_2d_image_to_pseudo_3d(frame)
            # 0000 suffix is necessary for nn-UNet
            niigz_path = target_path_images / (slice_id + "_" + str(ind) + "_0000.nii.gz")
            sitk.WriteImage(frame_2d, str(niigz_path))
    else:
        print("Skipping incomplete series or series with different anatomical plane, slice id: {}".format(slice_id))


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

    Returns
    -------
    sitk_image : SimpleITK.Image
       A merged image
    """

    frame_files_glob = frames_folder.glob(slice_full_id + "*.nii.gz")
    # Sort by file index to merge the images in the correct order
    # Scans have additional "_0000" suffix, masks do not have it
    sort_index = -1 if masks else -2
    files = sorted([file for file in frame_files_glob], key=lambda file: int(file.name[:-7].split("_")[sort_index]))

    image = []
    for frame_file in files:
        frame = sitk.ReadImage(str(frame_file))
        image.append(sitk.GetArrayFromImage(frame)[0])

    image = np.array(image).astype(np.uint8)
    sitk_image = sitk.GetImageFromArray(image)

    # Extract and assign metadata
    slice_metadata_path = metadata_path / (slice_full_id + ".json")
    with open(slice_metadata_path) as metadata_file:
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
    return sitk_image


# TODO: or move it to detection data split?
# Detection dataset
def extract_detection_dataset(slices, images_folder, target_folder):
    """
    Extracts a subset of cine-MRI slices for adhesion detection task
    Parameters
    ----------
    slices : list of CineMRISlice
       A list of slices to include into the detection dataset
    images_folder : Path
       A path to the images folder in the main archive
    target_folder : Path
       A destination path to save the dataset

    """
    for slice in slices:
        study_dir = target_folder / slice.patient_id / slice.study_id
        study_dir.mkdir(exist_ok=True, parents=True)
        slice_path = slice.build_path(images_folder)
        slice_target_path = slice.build_path(target_folder)
        shutil.copyfile(slice_path, slice_target_path)


def extract_detection_data(argv):
    """ Command line wrapper for the extract_detection_dataset method
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_file', type=str, required=True,
                        help="a path to a file with fill ids of positive slices")
    parser.add_argument('--negative_file', type=str, required=True,
                        help="a path to a file with fill ids of negative slices")
    parser.add_argument('--images', type=str, required=True, help="a path to image folder in the cine-MRI archive")
    parser.add_argument('--target_folder', type=str, required=True,
                        help="a path to a folder to place the detection subset")

    args = parser.parse_args(argv)

    positive_file_path = Path(args.positive_file)
    negative_file_path = Path(args.negative_file)
    images_path = Path(args.images)
    target_path = Path(args.target_folder)
    target_path.mkdir(parents=True)

    positive_slices = slices_from_full_ids_file(positive_file_path)
    negative_slices = slices_from_full_ids_file(negative_file_path)
    slices = positive_slices + negative_slices
    extract_detection_dataset(slices, images_path, target_path)


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(99)

    # Very first argument determines action
    actions = {
        "extract_detection_data": extract_detection_data,
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: registration ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])