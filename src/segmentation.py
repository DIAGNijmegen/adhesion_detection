#!/usr/local/bin/python3

# Functions related to abdominal cavity segmentation
import shutil
import sys
import subprocess
import argparse
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from cinemri.utils import get_patients
from data_extraction import extract_frames, merge_frames
from postprocessing import fill_in_holes
from utils import patients_from_full_ids
from data_extraction import save_frame

FRAMES_FOLDER = "frames"
MASKS_FOLDER = "masks"
METADATA_FOLDER = "images_metadata"
MERGED_MASKS_FOLDER = "merged_masks"

container_input_dir = Path("/tmp/nnunet/input")
container_output_dir = Path("/tmp/nnunet/output")


def _patients_from_args(args):
    # Extract patients with includes studies and slices to run inference for if file with slices IDs is given
    if args.slices_filter:
        with open(args.slices_filter) as file:
            lines = file.readlines()
            full_ids = [line.strip() for line in lines]
            patients = patients_from_full_ids(full_ids)
    else:
        patients = None

    return patients


def _find_segmented_frame_index(segmentation_path):
    """
    Finds the index of the frame on cine-MRI slice for which segmentation was made.
    In the last version of annotations it can be any frame and only one frame per slice is annotated
    Parameters
    ----------
    segmentation_path : Path
      A path to .mha file with annotation of abdominal cavity

    Returns
    -------
    index : int
       The index of the segmented frame
    """
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
                segmented_frame_index = _find_segmented_frame_index(mask_path)
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
    parser.add_argument('--masks', type=str, default="cavity_segmentations",
                        help="a name of the folder with cavity segmentations in the archive")
    parser.add_argument('--target_images', type=str, default="images",
                        help="a name of the images folder in the segmentation subset")
    parser.add_argument('--target_masks', type=str, default="masks",
                        help="a name of the folder with cavity segmentations in the segmentation subset")
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


def extract_complete_segmentation_data(images_path,
                                       target_path,
                                       target_frames_folder=FRAMES_FOLDER,
                                       target_metadata_folder=METADATA_FOLDER,
                                       patients=None):

    """
    Extracts frames for segmentation and the corresponding metadata from the whole cine-MRI archive
    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images
    target_path : Path
       A path where to save the extracted frames and metadata
    target_frames_folder : Path, default=FRAMES_FOLDER
       A subfolder of target_path to save the extracted frames
    target_metadata_folder : Path, default=METADATA_FOLDER
       A subfolder of target_path to save the metadata
    patients : list of Patient, optional
       A list of patients to extract the data for.
       If not provided the data are extracted for all patients at images_path
    """

    target_path.mkdir()

    target_images_path = target_path / target_frames_folder
    target_images_path.mkdir()

    target_metadata_path = target_path / target_metadata_folder
    target_metadata_path.mkdir()

    patients = patients if patients is not None else get_patients(images_path)
    for patient in patients:
        for cinemri_slice in patient.cinemri_slices:
            slice_path = cinemri_slice.build_path(images_path)
            slice = sitk.ReadImage(str(slice_path))
            extract_frames(slice, cinemri_slice.full_id, target_images_path, target_metadata_path)


def extract_complete_data(argv):
    """A command line wrapper of extract_complete_segmentation_data

    Parameters
    ----------
    argv: list of str
        Command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument("target_path", type=str, help="a path to save the extracted frames")
    parser.add_argument("--slices_filter", type=str, required=False, help="a path to a file with full id of slices "
                                                                          "which to extract the data for")
    args = parser.parse_args(argv)

    archive_path = Path(args.archive_path)
    target_path = Path(args.target_path)
    patients = _patients_from_args(args)
    extract_complete_segmentation_data(archive_path, target_path, patients=patients)


def delete_folder_contents(folder):
    """
    An auxiliary function to delete content of a folder
    Parameters
    ----------
    folder : str
       A path string of a folder which content to delete

    """
    for path in Path(folder).iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _predict_and_save(nnUNet_model_path,
                      nnUNet_input_dir,
                      nnUNet_output_dir,
                      output_path,
                      network,
                      task_id,
                      folds=None):
    """
    Runs inference with nn-UNet on the subset of input files and copies prediction to the specified location.
    When prediction is copies, nn-UNet input and output folder are emptied.
    Parameters
    ----------
    nnUNet_model_path : Path
       A path to nn-UNet model to run inference with
    nnUNet_input_dir : Path
       A path to a folder that contains images to run inference for
    nnUNet_output_dir : Path
       A path to a folder where to save nn-UNet prediction
    output_path : Path
       A path to a folder where to copy nn-UNet prediction
    network : str
       A type of nnU-Net network
    task_id : str
       An id of a task for nnU-Net
    folds : str, optional
       A string, specifying which folds to use for prediction, e.g "0,1,2"
    """
    cmd = [
        "nnunet", "predict", task_id,
        "--results", str(nnUNet_model_path),
        "--input", str(nnUNet_input_dir),
        "--output", str(nnUNet_output_dir),
        "--network", network
    ]

    print("First cmd {}".format(cmd))

    if folds:
        print("Adding folds")
        cmd.append('--folds')
        cmd.append(folds)

    print("Second cmd {}".format(cmd))

    subprocess.check_call(cmd)

    masks_files = container_output_dir.glob("*.nii.gz")
    for mask_path in masks_files:
        print("Saving a mask for {}".format(mask_path.name))
        shutil.copyfile(mask_path, output_path / mask_path.name)

    delete_folder_contents(nnUNet_input_dir)
    delete_folder_contents(nnUNet_output_dir)


# Currently shutil.copy() does not work on cluster because it is not allowed to change permissions of a file on a mount
# The workaround is to split input files in batches, run inference for each batch and then copy the prediction
# for a batch to Chansey. This way it less likely to lose the results if a job gets interrupted
def segment_abdominal_cavity(nnUNet_model_path,
                             input_path,
                             output_path,
                             task_id="Task101_AbdomenSegmentation",
                             network="2d",
                             batch_size=1000,
                             folds=None):
    """Runs inference of segmentation with the saved nnU-Net model

    Parameters
    ----------
    nnUNet_model_path : Path
       A path to the "results" folder generated during nnU-Net training
    input_path :  Path
       A path to a folder that contain the images to run inference for
    output_path : Path
       A path to a folder where to save the predicted segmentation
    task_id : str, default="Task101_AbdomenSegmentation"
       An id of a task for nnU-Net
    network : str, default="2d"
       A type of nnU-Net network
    batch_size : int, default=1000
       A number of images in a batch for a single prediction iteration
    folds : str, optional
       A string, specifying which folds to use for prediction, e.g "0,1,2"
    """

    # Create temporary input and output folders for nn-UNet inside the container
    container_input_dir.mkdir(exist_ok=True, parents=True)
    container_output_dir.mkdir(exist_ok=True, parents=True)

    output_path.mkdir(parents=True, exist_ok=True)

    print("Segmenting inspiration and expiration frames with nnU-Net")
    input_files = input_path.glob("*.nii.gz")
    for (index, input_frame_path) in enumerate(input_files):
        shutil.copy(input_frame_path, container_input_dir)
        if index > 0 and index % batch_size == 0:
            _predict_and_save(nnUNet_model_path,
                              container_input_dir,
                              container_output_dir,
                              output_path,
                              network,
                              task_id,
                              folds)

    # Process remaining images
    _predict_and_save(nnUNet_model_path,
                      container_input_dir,
                      container_output_dir,
                      output_path,
                      network,
                      task_id,
                      folds)

    print("Running post processing")
    fill_in_holes(output_path)


def segment(argv):
    """A command line wrapper of segment_abdominal_cavity

    Parameters
    ----------
    argv: list of str
       Command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="a path to the folder which contains a nnUNet input")
    parser.add_argument('--output', type=str, help="a path to the folder to save a nnUNet output")
    parser.add_argument("--nnUNet_results", type=str, required=True,
                        help="a path to the \"results\" folder generated during nnU-Net training")
    parser.add_argument("--task", type=str, default="Task101_AbdomenSegmentation", help="an id of a task for nnU-Net")
    parser.add_argument('--folds', type=str, required=False, help="folds which use for prediction")

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    nnUNet_model_path = Path(args.nnUNet_results)
    task_id = args.task
    folds = args.folds if args.folds else None
    print("Folds {}".format(folds))
    segment_abdominal_cavity(nnUNet_model_path, input_path, output_path, task_id, folds=folds)


def merge_segmentation(segmentation_path,
                       metadata_path,
                       target_path):
    """
    Merges segmentation masks saved as separate frames into 3D images corresponding to cine-MRI slices
    Parameters
    ----------
    segmentation_path : Path
       A path to segmentation masks
    metadata_path : Path
       A path to cine-MRI slices metadata
    target_path : Path
       A path where to save merged 3D cine-MRI slices masks

    Returns
    -------

    """
    target_path.mkdir(exist_ok=True)

    # Get slices ids from metadata folder
    slices_metadata_glob = metadata_path.glob("*.json")
    slices_full_ids = [slice_file.name[:-5] for slice_file in slices_metadata_glob]
    patients = patients_from_full_ids(slices_full_ids)

    # Create folders hierarchy matching hierarchy of images in the archive: patientID -> studyID
    # and merge and save each slice
    for patient in patients:
        patient.build_path(target_path).mkdir()
        for study in patient.studies:
            study_path = study.build_path(target_path)
            study_path.mkdir()

            for slice in study.slices:
                # Extract frames matching the slice id and its metadata, merge into .mha
                # and save in a study folder
                merge_frames(slice.full_id, segmentation_path, study_path, metadata_path)


def merge(argv):
    """A command line wrapper of merge_segmentation

    Parameters
    ----------
    argv: list of str
        Command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("segmentation_path", type=str, help="a path to the predicted segmentation")
    parser.add_argument("--metadata_path", type=str, help="a path to the images metadata", required=True)
    parser.add_argument("--target_path", type=str, help="a path to save the merged prediction", required=True)
    args = parser.parse_args(argv)

    segmentation_path = Path(args.segmentation_path)
    metadata_path = Path(args.metadata_path)
    target_path = Path(args.target_path)
    merge_segmentation(segmentation_path, metadata_path, target_path)


def run_full_inference(images_path,
                       output_path,
                       nnUNet_model_path,
                       nnUNet_task,
                       patients=None,
                       folds=None):
    """
    Runs inference of segmentation masks for the whole cine-MRI archive with nn-UNet

    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images
    output_path : Path
       A path where to save nn-UNet prediction. After execution of the function
       output_path / FRAMES_FOLDER will contain separate frames of all cine-MRI slices in the archive
       output_path / METADATA_FOLDER will contain metadata of all cine-MRI slices in the archive
       output_path / MASKS_FOLDER will contain masks predicted for all frames located in FRAMES_FOLDER
       output_path / MERGED_MASKS_FOLDER will contain merged masks corresponding to all cine-MRI slices in the archive
       placed with the preserved archive folders hierarchy
    nnUNet_model_path : Path
       A path where nn-UNet model to be used is located
    nnUNet_task : str
       An id of a task for nnU-Net
    patients : list of Patient, optional
       A list of patients with studies and slices to run inference for
    folds : str, optional
       A string, specifying which folds to use for prediction, e.g "0,1,2"
    """

    extract_complete_segmentation_data(images_path, output_path, patients=patients)

    nnUNet_input_path = output_path / FRAMES_FOLDER
    nnUNet_output_path = output_path / MASKS_FOLDER

    segment_abdominal_cavity(nnUNet_model_path,
                             nnUNet_input_path,
                             nnUNet_output_path,
                             nnUNet_task,
                             folds=folds)

    metadata_path = output_path / METADATA_FOLDER
    merged_masks_path = output_path / MERGED_MASKS_FOLDER
    merge_segmentation(nnUNet_output_path, metadata_path, merged_masks_path)


def full_inference(argv):
    """A command line wrapper of run_full_inference

    Parameters
    ----------
    argv: list of str
        Command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("images_path", type=str, help="a path to a folder with cine-MRI images")
    parser.add_argument("--slices_filter", type=str, required=False, help="a path to a file with full id of slices "
                                                                          "which to run the inference for")
    parser.add_argument("--output_path", type=str, help="a path where to save the output")
    parser.add_argument("--nnUNet_results", type=str, required=True,
                        help="a path to the \"results\" folder generated during nnU-Net training")
    parser.add_argument("--task", type=str, default="Task101_AbdomenSegmentation", help="an id of a task for nnU-Net")
    parser.add_argument('--folds', type=str, required=False, help="folds which use for prediction")
    args = parser.parse_args(argv)

    images_path = Path(args.images_path)
    output_path = Path(args.output_path)
    nnUNet_results_path = Path(args.nnUNet_results)
    nnUNet_task = args.task
    folds = args.folds if args.folds else None
    patients = _patients_from_args(args)

    run_full_inference(images_path, output_path, nnUNet_results_path, nnUNet_task, patients, folds)


if __name__ == '__main__':

    actions = {
        "extract_segmentation" : extract_segmentation,
        "extract_complete_data": extract_complete_segmentation_data,
        "segment": segment,
        "merge": merge,
        "full_inference": full_inference,
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: data_conversion ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
