#!/usr/local/bin/python3

import sys
import pickle
import argparse
import json
import subprocess
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from cinemri.utils import get_patients, Patient
from data_conversion import convert_2d_image_to_pseudo_3d
from visceral_slide import VisceralSlideDetector
import matplotlib.pyplot as plt
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, TRAIN_TEST_SPLIT_FILE_NAME, TRAIN_PATIENTS_KEY,\
    TEST_PATIENTS_KEY
from postprocessing import fill_in_holes

# TODO:
# How do we evaluate registration?
NNUNET_INPUT_FOLDER = "nnUNet_input"
PREDICTED_MASKS_FOLDER = "nnUNet_masks"
RESULTS_FOLDER = "visceral_slide"


def extract_insp_exp_frames(images_path,
                            patients_ids,
                            inspexp_file_path,
                            destination_path):
    """Extracts inspiration and expiration frames and saves them to nn-UNet input format.

    The file names of the extracted frames have the following structure:
    patientId_scanId_sliceId_[insp/exp]_0000.nii.gz
    This helps to recover folders hierarchy when the calculated visceral slide for each slice is being saved.

    Parameters
    ----------
    images_path : Path
       A path to a folder in the archive that contains cine-MRI scans
    patients_ids : list of str
       A list of the patients ids for which frames will be extracted
    inspexp_file_path : Path
       A path to a json file that contains information about inspiration and expiration frames
       for each slice
    destination_path : Path
       A path where to save the extracted inspiration and expiration frames in nn-UNet input format
    """

    destination_path.mkdir(exist_ok=True)
    print("Starting extraction of inspiration and expiration frames")
    print("The following subset of patients is considered:")
    print(patients_ids)
    print("The frames will be stored in {}".format(str(destination_path)))

    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # Extract the subset of patients
    patients = get_patients(images_path)
    patients = [patient for patient in patients if patient.id in patients_ids]

    for patient in patients:
        print("Extracting frames for slices of a patient {}".format(patient.id))
        if patient.id in inspexp_data:
            patient_data = inspexp_data[patient.id]
        else:
            print("No data about inspitation and expiration frames for a patient {}".format(patient.id))
            continue

        for (scan_id, slices) in patient.scans.items():
            if scan_id in patient_data:
                scan_data = patient_data[scan_id]
            else:
                print("No data about inspiration and expiration frames for a scan {}".format(scan_id))
                continue

            for slice in slices:
                slice_stem = slice[:-4]
                if slice_stem in scan_data:
                    inspexp_frames = scan_data[slice_stem]
                else:
                    print("No data about inspiration and expiration frames for a slice {}".format(slice_stem))
                    continue

                separator = "_"
                slice_id = separator.join([patient.id, scan_id, slice_stem])

                slice_path = images_path / patient.id / scan_id / slice
                slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

                # Extract and save inspiration frame
                insp_frame = slice_array[inspexp_frames[0]]
                insp_pseudo_3d_image = convert_2d_image_to_pseudo_3d(insp_frame)
                insp_file_path = destination_path / (slice_id + "_insp_0000.nii.gz")
                sitk.WriteImage(insp_pseudo_3d_image, str(insp_file_path))

                # Extract and save expiration frame
                exp_frame = slice_array[inspexp_frames[1]]
                exp_pseudo_3d_image = convert_2d_image_to_pseudo_3d(exp_frame)
                exp_file_path = destination_path / (slice_id + "_exp_0000.nii.gz")
                sitk.WriteImage(exp_pseudo_3d_image, str(exp_file_path))


def extract_insp_exp(argv):
    """A command line wrapper of extract_segmentation_data

    Parameters
    ----------
    argv: list of str
       Command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument("--output", type=str, required=True,
                        help="a path to save the extracted inspiration and expiration frames")
    parser.add_argument("--mode", type=str, default="train",
                        help="indicates which subset of patients to select, can be \"train\" or \"test\"")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    images_path = data_path / IMAGES_FOLDER
    inspexp_file_path = data_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    output_path = Path(args.output)
    destination_path = output_path / NNUNET_INPUT_FOLDER
    mode = args.mode

    if mode == "train":
        patients_key = TRAIN_PATIENTS_KEY
    elif mode == "test":
        patients_key = TEST_PATIENTS_KEY
    else:
        raise ValueError("Usuppotred mode: should be train or test")

    train_test_split_file_path = data_path / METADATA_FOLDER / TRAIN_TEST_SPLIT_FILE_NAME
    with open(train_test_split_file_path) as train_test_split_file:
        train_test_split = json.load(train_test_split_file)
    patients_ids = train_test_split[patients_key]

    # Create output folder
    output_path.mkdir(parents=True)
    extract_insp_exp_frames(images_path, patients_ids, inspexp_file_path, destination_path)


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

    cmd = [
        "nnunet", "predict", task_id,
        "--results", nnUNet_model_path,
        "--input", input_path,
        "--output", output_path,
        "--network", network
    ]

    print("Segmenting inspiration and expiration frames with nnU-Net")
    subprocess.check_call(cmd)

    # Postprocess the prediction by filling in holes
    fill_in_holes(Path(output_path))


def segment(argv):
    """A command line wrapper of segment_abdominal_cavity

    Parameters
    ----------
    argv: list of str
       Command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=str,
                        help="a path to the folder which contains a folder with nnU-Net input and "
                             "where a folder for saving the predicted segmentation will be created")
    parser.add_argument("--nnUNet_results", type=str, required=True,
                        help="a path to the \"results\" folder generated during nnU-Net training")
    parser.add_argument("--task", type=str, default="Task101_AbdomenSegmentation", help="an id of a task for nnU-Net")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    nnUNet_model_path = args.nnUNet_results
    task_id = args.task
    input_path = data_path / NNUNET_INPUT_FOLDER
    output_path = data_path / PREDICTED_MASKS_FOLDER
    segment_abdominal_cavity(nnUNet_model_path, str(input_path), str(output_path), task_id)


def compute_visceral_slide(images_path,
                           masks_path,
                           target_path):
    """Computes visceral slide for each slice with VisceralSlideDetector

    Parameters
    ----------
    images_path : Path
       A path to a folder containing inspiration and expiration frames of slices in .nii.gz
    masks_path : Path
       A path to a folder containing predicted masks for inspiration and expiration
                       frames of slices in .nii.gz
    target_path : Path
       A path to a folder to save visceral slide
    Returns
    -------

    """

    target_path.mkdir(exist_ok=True)
    print("Computing visceral slide for each slice")
    print("The results will be stored in {}".format(str(target_path)))
    visceral_slide_detector = VisceralSlideDetector()

    slices_glob = masks_path.glob("*insp.nii.gz")
    slices_id_chunks = [f.name[:-12].split("_") for f in slices_glob]
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
                slice_path = scan_path / slice_id
                slice_path.mkdir()

                separator = "_"
                slice_name = separator.join([patient.id, scan_id, slice_id])
                print("Processing a slice {}".format(slice_name))

                insp_path = images_path / (slice_name + "_insp_0000.nii.gz")
                insp_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(insp_path)))[0].astype(np.uint32)
                insp_mask_path = masks_path / (slice_name + "_insp.nii.gz")
                insp_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(insp_mask_path)))[0]

                exp_path = images_path / (slice_name + "_exp_0000.nii.gz")
                exp_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(exp_path)))[0].astype(np.uint32)
                exp_mask_path = masks_path / (slice_name + "_exp.nii.gz")
                exp_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(exp_mask_path)))[0]

                x, y, visceral_slide = visceral_slide_detector.get_visceral_slide(exp_frame, exp_mask, insp_frame, insp_mask)

                # Save pickle and figure
                pickle_path = slice_path / "visceral_slide.pkl"
                slide_dict = {"x": x, "y": y, "slide": visceral_slide}
                with open(pickle_path, "w+b") as file:
                    pickle.dump(slide_dict, file)

                color_matrix = np.zeros((len(x), 4))
                slide_normalized = np.abs(visceral_slide) / np.max(np.abs(visceral_slide))
                for i in range(len(x)):
                    color_matrix[i, 0] = 1
                    color_matrix[i, 3] = slide_normalized[i]

                plt.figure()
                plt.imshow(exp_frame, cmap="gray")
                plt.scatter(x, y, c=color_matrix)
                plt.axis('off')
                plt.savefig(slice_path / "visceral_slide_overlayed.png", bbox_inches='tight', pad_inches=0)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, exp_frame.shape[0] - y, c=color_matrix)
                ax.set_aspect(1)
                plt.xlim((0, exp_frame.shape[1]))
                plt.ylim((0, exp_frame.shape[0]))
                plt.axis('off')
                fig.savefig(slice_path / "visceral_slide.png", bbox_inches='tight', pad_inches=0)


def compute(argv):
    """A command line wrapper of compute_visceral_slide

    Parameters
    ----------
    argv: list of str
       Command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=str,
                        help="a path to the folder which contains folders with images and masks for visceral slide "
                             "computation and in which the computed visceral slide will be saved")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    images_path = data_path / NNUNET_INPUT_FOLDER
    masks_path = data_path / PREDICTED_MASKS_FOLDER
    output_path = data_path / RESULTS_FOLDER

    compute_visceral_slide(images_path, masks_path, output_path)


def run_pileline(data_path,
                 nnUNet_model_path,
                 output_path,
                 patients_set_key,
                 task_id="Task101_AbdomenSegmentation"):
    """Runs the pipeline to compute visceral slide for all scans of the specified set of the patients.

    The pipeline consists of the following steps:
    - Extraction of inspiration and expiration frames
    - Segmentation of abdominal cavity on these fames with nn-UNet
    - Calculation of visceral slide along the abdominal cavity contour

    Parameters
    ----------
    data_path : Path
       A path to the full cine-MRI data archive
    nnUNet_model_path : Path
       A path to the "results" folder generated during nnU-Net training
    output_path : Path
       A path to a folder to save visceral slide
    patients_set_key : str
       A key indicating which subset of patients to select, can be "train" or "test"
    task_id : str, default="Task101_AbdomenSegmentation"
       An id of a task for nnU-Net
    """

    # Create output folder
    output_path.mkdir(parents=True)

    # Extract inspiration and expiration frames and save in nnU-Net input format
    images_path = data_path / IMAGES_FOLDER
    # Extract a subset of patients
    train_test_split_file_path = data_path / METADATA_FOLDER / TRAIN_TEST_SPLIT_FILE_NAME
    with open(train_test_split_file_path) as train_test_split_file:
        train_test_split = json.load(train_test_split_file)
    patients_ids = train_test_split[patients_set_key]
    inspexp_file_path = data_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    nnUNet_input_path = output_path / NNUNET_INPUT_FOLDER
    extract_insp_exp_frames(images_path, patients_ids, inspexp_file_path, nnUNet_input_path)

    # Run inference with nnU-Net
    nnUNet_output_path = output_path / PREDICTED_MASKS_FOLDER
    segment_abdominal_cavity(str(nnUNet_model_path), str(nnUNet_input_path), str(nnUNet_output_path), task_id)

    # Compute visceral slide with nnU-Net segmentation masks
    visceral_slide_path = output_path / RESULTS_FOLDER
    compute_visceral_slide(nnUNet_input_path, nnUNet_output_path, visceral_slide_path)
    print("Done")


def pipeline(argv):
    """A command line wrapper of run_pileline

    Parameters
    ----------
    argv: list of str
        Command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument('model', type=str, help="a path to the \"results\" folder generated during nnU-Net training")
    parser.add_argument('--output', type=str, required=True, help="a path to a folder to save visceral slide")
    parser.add_argument('--task', type=str, default="Task101_AbdomenSegmentation", help="an id of a task for nnU-Net")
    parser.add_argument('--mode', type=str, default="train",
                        help="indicates which subset of patients to select, can be \"train\" or \"test\"")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    nnUNet_results_path = Path(args.model)
    output_path = Path(args.output)
    task = args.task
    mode = args.mode

    if mode == "train":
        run_pileline(data_path, nnUNet_results_path, output_path, TRAIN_PATIENTS_KEY, task)
    elif mode == "test":
        run_pileline(data_path, nnUNet_results_path, output_path, TEST_PATIENTS_KEY, task)
    else:
        raise ValueError("Usupported mode: should be train or test")


if __name__ == '__main__':

    # Very first argument determines action
    actions = {
        "extract_frames": extract_insp_exp,
        "segment_frames": segment,
        "compute": compute,
        "pipeline": pipeline
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: registration ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
