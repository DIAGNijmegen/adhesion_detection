#!/usr/local/bin/python3

import os
import shutil
import sys
import subprocess
import argparse
import numpy as np
from pathlib import Path
from cinemri.utils import get_patients, Patient
from config import IMAGES_FOLDER
from data_conversion import extract_frames, merge_frames
from postprocessing import fill_in_holes

SEPARATOR = "_"
FRAMES_FOLDER = "frames"
MASKS_FOLDER = "masks"
METADATA_FOLDER = "images_metadata"
MERGED_MASKS_FOLDER = "merged_masks"

container_input_dir = Path("/tmp/nnunet/input")
container_output_dir = Path("/tmp/nnunet/output")

# function to extract frames and metadata
def extract_segmentation_data(archive_path,
                              target_path,
                              target_frames_folder=FRAMES_FOLDER,
                              target_metadata_folder=METADATA_FOLDER,
                              images_folder=IMAGES_FOLDER):

    target_path.mkdir()

    target_images_path = target_path / target_frames_folder
    target_images_path.mkdir()

    target_metadata_path = target_path / target_metadata_folder
    target_metadata_path.mkdir()

    patients = get_patients(archive_path)
    for patient in patients:
        for (scan_id, slices) in patient.scans.items():
            for slice in slices:
                slice_path = archive_path / images_folder / patient.id / scan_id / slice
                slice_id = SEPARATOR.join([patient.id, scan_id, slice[:-4]])
                extract_frames(slice_path, slice_id, target_images_path, target_metadata_path)


def extract_data(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument("target_path", type=str, help="a path to save the extracted frames")
    args = parser.parse_args(argv)

    archive_path = Path(args.archive_path)
    target_path = Path(args.target_path)
    extract_segmentation_data(archive_path, target_path)


def delete_folder_contents(folder):
    for path in Path(folder).iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _predict_and_save(nnUNet_model_path, output_path, network, task_id):
    cmd = [
        "nnunet", "predict", task_id,
        "--results", nnUNet_model_path,
        "--input", str(container_input_dir),
        "--output", str(container_output_dir),
        "--network", network
    ]
    subprocess.check_call(cmd)

    masks_files = container_output_dir.glob("*.nii.gz")
    for mask_path in masks_files:
        print("Saving a mask for {}".format(mask_path.name))
        shutil.copyfile(mask_path, output_path / mask_path.name)

    delete_folder_contents(container_input_dir)
    delete_folder_contents(container_output_dir)


# Currently shutil.copy() does not work on cluster because it is not allowed to change permissions of a file on a mount
# Workaround is to iterate through input files and separately for each file run inference and move it to
# the destination path on Chansey. This way the progress is not lost if the jobs is interrupted
def segment_abdominal_cavity(nnUNet_model_path,
                             input_path,
                             output_path,
                             task_id="Task101_AbdomenSegmentation",
                             network="2d"):
    """Runs inference of segmentation with the saved nnU-Net model

    Parameters
    ----------
    nnUNet_model_path : str
       A path to the "results" folder generated during nnU-Net training
    input_path :  str
       A path to a folder that contain the images to run inference for
    output_path : str
       A path to a folder where to save the predicted segmentation
    task_id : str, default="Task101_AbdomenSegmentation"
       An id of a task for nnU-Net
    network : str, default="2d"
       A type of nnU-Net network
    """

    container_input_dir.mkdir(exist_ok=True, parents=True)
    container_output_dir.mkdir(exist_ok=True, parents=True)

    output_path = Path(output_path)
    # Prepare output directory to prevent crashes
    output_path.mkdir(parents=True, exist_ok=True)

    print("Segmenting inspiration and expiration frames with nnU-Net")
    input_path = Path(input_path)
    input_files = input_path.glob("*.nii.gz")
    batch_size = 1000
    for (index, input_frame_path) in enumerate(input_files):
        shutil.copy(input_frame_path, container_input_dir)
        if index > 0 and index % batch_size == 0:
            _predict_and_save(nnUNet_model_path, output_path, network, task_id)

    # Process remaining images
    _predict_and_save(nnUNet_model_path, output_path, network, task_id)

    print("Running post processing")
    fill_in_holes(Path(output_path))


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
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    nnUNet_model_path = args.nnUNet_results
    task_id = args.task
    segment_abdominal_cavity(nnUNet_model_path, str(input_path), str(output_path), task_id)


def merge_segmentation(segmentation_path,
                       metadata_path,
                       target_path):
    target_path.mkdir(exist_ok=True)

    # Get slices ids from metadata folder
    slices_metadata_glob = metadata_path.glob("*.json")
    slices_id_chunks = [f.name[:-5].split("_") for f in slices_metadata_glob]
    slices_id_chunks = np.array(slices_id_chunks)

    # Extract patients to recover original folder structure
    patient_ids = np.unique(slices_id_chunks[:, 0])
    patients = []
    for patient_id in patient_ids:
        patient = Patient(patient_id)
        slices = slices_id_chunks[slices_id_chunks[:, 0] == patient_id]
        for _, scan_id, slice_id in slices:
            patient.add_slice(slice_id, scan_id)

        patients.append(patient)

    for patient in patients:
        patient_path = target_path / patient.id
        patient_path.mkdir()
        for scan_id in patient.scan_ids:
            scan_path = patient_path / scan_id
            scan_path.mkdir()

            for slice_id in patient.scans[scan_id]:
                slice_name = SEPARATOR.join([patient.id, scan_id, slice_id])
                slice_metadata_path = metadata_path / (slice_name + ".json")

                # Extract frames matching the slice id and its metadata, merge into .mha
                # and save in a scan folder
                slice_glob_pattern = slice_name + "*.nii.gz"
                merge_frames(slice_glob_pattern, slice_id, segmentation_path, scan_path, slice_metadata_path)


def merge(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("segmentation_path", type=str, help="a path to the predicted segmentation")
    parser.add_argument("--metadata_path", type=str, help="a path to the images metadata", required=True)
    parser.add_argument("--target_path", type=str, help="a path to save the merged prediction", required=True)
    args = parser.parse_args(argv)

    segmentation_path = Path(args.segmentation_path)
    metadata_path = Path(args.metadata_path)
    target_path = Path(args.target_path)
    merge_segmentation(segmentation_path, metadata_path, target_path)


def run_full_inference(archive_path,
                       output_path,
                       nnUNet_model_path,
                       nnUNet_task):

    extract_segmentation_data(archive_path, output_path)

    nnUNet_input_path = output_path / FRAMES_FOLDER
    nnUNet_output_path = output_path / MASKS_FOLDER

    segment_abdominal_cavity(nnUNet_model_path,
                             nnUNet_input_path,
                             nnUNet_output_path,
                             nnUNet_task)

    metadata_path = output_path / METADATA_FOLDER
    merged_masks_path = output_path / MERGED_MASKS_FOLDER
    merge_segmentation(nnUNet_output_path, metadata_path, merged_masks_path)


def full_inference(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path", type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument("--output_path", type=str, help="a path where to save the output")
    parser.add_argument("--nnUNet_results", type=str, required=True,
                        help="a path to the \"results\" folder generated during nnU-Net training")
    parser.add_argument("--task", type=str, default="Task101_AbdomenSegmentation", help="an id of a task for nnU-Net")
    args = parser.parse_args(argv)

    archive_path = Path(args.archive_path)
    output_path = Path(args.output_path)
    nnUNet_results_path = Path(args.nnUNet_results)
    nnUNet_task = args.task

    run_full_inference(archive_path, output_path, nnUNet_results_path, nnUNet_task)


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate")
    target_path_images = Path("../../data/cinemri_mha/frames")
    target_path_metadata = Path("../../data/images_metadata")
    segmentation_path = Path("../../data/masks")
    merged_segmentation_path = Path("../../data/merged_segmentation1")

    #extract_segmentation_data(archive_path, target_path_images)
    #simulate_segmentation(target_path_images, segmentation_path)
    merge_segmentation(segmentation_path, target_path_metadata, merged_segmentation_path)


if __name__ == '__main__':
    test()

    """
    actions = {
        "extract_data": extract_data,
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
    """
