#!/usr/local/bin/python3

import sys
import pickle
import argparse
import json
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from cinemri.utils import get_patients, Patient
from data_conversion import convert_2d_image_to_pseudo_3d
from visceral_slide import VisceralSlideDetector, CumulativeVisceralSlideDetector
import matplotlib.pyplot as plt
from cinemri.config import ARCHIVE_PATH
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, TRAIN_TEST_SPLIT_FILE_NAME, TRAIN_PATIENTS_KEY,\
    TEST_PATIENTS_KEY, VISCERAL_SLIDE_FILE
from segmentation import segment_abdominal_cavity
from utils import slices_from_full_ids_file

# TODO: maybe in future rewrite the pipeline to extract the full segmentation needed for cumulative visceral slide
# TODO: How do we evaluate registration?
NNUNET_INPUT_FOLDER = "nnUNet_input"
PREDICTED_MASKS_FOLDER = "nnUNet_masks"
RESULTS_FOLDER = "visceral_slide"


# TODO: probably also add an option to load saved segmentation since it I run it for the whole dataset
def get_patients_ids(train_test_split, mode):
    """
    Filters patients ids based on the data split for nn-UNet training and mode ("all", "train", "test")
    Parameters
    ----------
    train_test_split : dict
       A dictionary containing train/test split used for nn-UNet training and evaluation
    mode : str
       A string id indicating which subset of patients to keep

    Returns
    -------
    patients_ids : list
       A list of patient ids
    """

    if mode == "all":
        patients_ids = train_test_split[TRAIN_PATIENTS_KEY] + train_test_split[TEST_PATIENTS_KEY]
    elif mode == "train":
        patients_ids = train_test_split[TRAIN_PATIENTS_KEY]
    elif mode == "test":
        patients_ids = train_test_split[TEST_PATIENTS_KEY]
    else:
        raise ValueError("Usuppotred mode: should be train or test")

    return patients_ids


def load_visceral_slide(visceral_slide_path):
    """
    Load the saved visceral slide
    Parameters
    ----------
    visceral_slide_path : Path
       A path to the file with saved visceral slide

    Returns
    -------
    x, y, visceral_slide : list of int, list of int, list of float
       Coordinates of abominal cavity and the corresponding values of visceral slide
    """

    visceral_slide_file_path = visceral_slide_path / VISCERAL_SLIDE_FILE

    with open(str(visceral_slide_file_path), "r+b") as file:
        visceral_slide_data = pickle.load(file)
        x, y = visceral_slide_data["x"], visceral_slide_data["y"]
        visceral_slide = visceral_slide_data["slide"]

    return x, y, visceral_slide


# TODO: should be wrapped into try/cath
def get_insp_exp_indices(slice, inspexp_data):
    """
    Loads indexes of inspiration and expiration frames for the specified cine-MRI slice
    Parameters
    ----------
    slice: CineMRISlice
        A cine-MRI slice for which to extract inspiration and expiration frames
    inspexp_data : dict
        A dictionary with inspiration / expiration frames data

    Returns
    -------
    insp_ind, exp_ind : ndarray
        The inspiration and expiration frames indexes
    """

    patient_data = inspexp_data[slice.patient_id]
    scan_data = patient_data[slice.scan_id]
    inspexp_frames = scan_data[slice.slice_id]
    insp_ind = inspexp_frames[0]
    exp_ind = inspexp_frames[1]
    return insp_ind, exp_ind


# TODO: should be wrapped into try/cath
def get_inspexp_frames(slice, inspexp_data, images_path):
    """
    Loads inspiration and expiration frames for the specified cine-MRI slice
    Parameters
    ----------
    slice: CineMRISlice
        A cine-MRI slice for which to extract inspiration and expiration frames
    inspexp_data : dict
       A dictionary with inspiration / expiration frames data
    images_path : Path
       A path to the image folder in cine-MRI archive

    Returns
    -------
    insp_frame, exp_frame : ndarray
       The inspiration and expiration frames
    """

    insp_ind, exp_ind = get_insp_exp_indices(slice, inspexp_data)

    # Load the expiration frame (visceral slide is computed for the expiration frame)
    slice_path = slice.build_path(images_path)
    insp_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))[insp_ind]
    exp_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))[exp_ind]
    return insp_frame, exp_frame


def get_insp_exp_frames_and_masks(slice, insp_ind, exp_ind, images_path, masks_path):
    """
    Loads inspiration and expiration frames of the slice and the corresponding mask
    Parameters
    ----------
    slice : CineMRISlice
        A cine-MRI slice for which to extract inspiration and expiration frames
    insp_ind, exp_ind : int
       Inspiration and expiration indices
    images_path, masks_path : Path
       Paths to images and masks

    Returns
    -------
    insp_frame, insp_mask, exp_frame, exp_mask : ndarray
       Inspiration frame and mask, expiration frame and mask

    """
    # load image
    slice_path = slice.build_path(images_path)
    slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
    insp_frame = slice_array[insp_ind].astype(np.uint32)
    exp_frame = slice_array[exp_ind].astype(np.uint32)

    # load mask
    mask_path = slice.build_path(masks_path)
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    insp_mask = mask_array[insp_ind]
    exp_mask = mask_array[exp_ind]

    return insp_frame, insp_mask, exp_frame, exp_mask


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
       A path to a folder with cine-MRI images
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

        for cinemri_slice in patient.cinemri_slices:
            if cinemri_slice.scan_id in patient_data:
                scan_data = patient_data[cinemri_slice.scan_id]

                if cinemri_slice.slice_id in scan_data:
                    inspexp_frames = scan_data[cinemri_slice.slice_id]
                else:
                    print("No data about inspiration and expiration frames for a slice {}".format(cinemri_slice.slice_id))
                    continue
            else:
                print("No data about inspiration and expiration frames for a scan {}".format(cinemri_slice.scan_id))
                continue

            slice_path = cinemri_slice.build_path(images_path)
            slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

            # Extract and save inspiration frame
            insp_frame = slice_array[inspexp_frames[0]]
            insp_pseudo_3d_image = convert_2d_image_to_pseudo_3d(insp_frame)
            insp_file_path = destination_path / (cinemri_slice.full_id + "_insp_0000.nii.gz")
            sitk.WriteImage(insp_pseudo_3d_image, str(insp_file_path))

            # Extract and save expiration frame
            exp_frame = slice_array[inspexp_frames[1]]
            exp_pseudo_3d_image = convert_2d_image_to_pseudo_3d(exp_frame)
            exp_file_path = destination_path / (cinemri_slice.full_id + "_exp_0000.nii.gz")
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
                        help="indicates which subset of patients to select, can be \"all\", \"train\" or \"test\"")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    images_path = data_path / IMAGES_FOLDER
    inspexp_file_path = data_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    output_path = Path(args.output)
    destination_path = output_path / NNUNET_INPUT_FOLDER
    mode = args.mode

    train_test_split_file_path = data_path / METADATA_FOLDER / TRAIN_TEST_SPLIT_FILE_NAME
    with open(train_test_split_file_path) as train_test_split_file:
        train_test_split = json.load(train_test_split_file)

    patients_ids = get_patients_ids(train_test_split, mode)

    # Create output folder
    output_path.mkdir(parents=True)
    extract_insp_exp_frames(images_path, patients_ids, inspexp_file_path, destination_path)


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

                x, y, visceral_slide = visceral_slide_detector.get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask)

                # Save pickle and figure
                pickle_path = slice_path / VISCERAL_SLIDE_FILE
                slide_dict = {"x": x, "y": y, "slide": visceral_slide}
                with open(pickle_path, "w+b") as file:
                    pickle.dump(slide_dict, file)

                slide_normalized = np.abs(visceral_slide) / np.abs(visceral_slide).max()

                plt.figure()
                plt.imshow(insp_frame, cmap="gray")
                plt.scatter(x, y, s=5, c=slide_normalized, cmap="jet")
                plt.axis('off')
                plt.savefig(slice_path / "visceral_slide_overlayed.png", bbox_inches='tight', pad_inches=0)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, insp_frame.shape[0] - y, s=5, c=slide_normalized, cmap="jet")
                ax.set_aspect(1)
                plt.xlim((0, insp_frame.shape[1]))
                plt.ylim((0, insp_frame.shape[0]))
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

    data_path = Path(args.work_dir)
    images_path = data_path / NNUNET_INPUT_FOLDER
    masks_path = data_path / PREDICTED_MASKS_FOLDER
    output_path = data_path / RESULTS_FOLDER

    compute_visceral_slide(images_path, masks_path, output_path)


def run_pileline(data_path,
                 nnUNet_model_path,
                 output_path,
                 mode,
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
    mode : str
       A key indicating which subset of patients to select, can be "all", "train" or "test"
    task_id : str, default="Task101_AbdomenSegmentation"
       An id of a task for nnU-Net
    """

    # Create output folder
    output_path.mkdir(parents=True)

    # Extract a subset of patients
    train_test_split_file_path = data_path / METADATA_FOLDER / TRAIN_TEST_SPLIT_FILE_NAME
    with open(train_test_split_file_path) as train_test_split_file:
        train_test_split = json.load(train_test_split_file)

    patients_ids = get_patients_ids(train_test_split, mode)

    images_folder = data_path / IMAGES_FOLDER
    inspexp_file_path = data_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    nnUNet_input_path = output_path / NNUNET_INPUT_FOLDER

    # Extract inspiration and expiration frames and save in nnU-Net input format
    extract_insp_exp_frames(images_folder, patients_ids, inspexp_file_path, nnUNet_input_path)

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
                        help="indicates which subset of patients to select, can be \"all\", \"train\" or \"test\"")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    nnUNet_results_path = Path(args.model)
    output_path = Path(args.output)
    task = args.task
    mode = args.mode

    run_pileline(data_path, nnUNet_results_path, output_path, mode, task)


def compute_cumulative_visceral_slide(images_path, masks_path, slices, output_path, series_stat=False, rest_def=False):
    """
    Computes and save cumulative visceral slide for the specified subset of slices
    Parameters
    ----------
    images_path : Path
       A path to image folder in the cine-MRI archive
    masks_path : Path
       A path to segmentation masks of the whole slices
    slices : list of CineMRISlice
       A list of CineMRISlice instances to compute visceral slice for
    output_path : Path
       A path where to save computed visceral slide
    """

    output_path.mkdir()
    visceral_slide_detector = CumulativeVisceralSlideDetector()

    for slice in slices:
        print("Computing cumulative visceral slide for the slice {}".format(slice.full_id))

        slice_output_path = slice.build_path(output_path, extension="")
        slice_output_path.mkdir(parents=True)

        slice_path = slice.build_path(images_path)
        slice_image = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

        mask_path = slice.build_path(masks_path)
        mask_image = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

        x, y, visceral_slide = visceral_slide_detector.get_visceral_slide(slice_image, mask_image, series_stat, rest_def)

        pickle_path = slice_output_path / VISCERAL_SLIDE_FILE
        slide_dict = {"x": x, "y": y, "slide": visceral_slide}
        with open(pickle_path, "w+b") as file:
            pickle.dump(slide_dict, file)


def cumulative_visceral_slide(argv):
    """A command line wrapper of compute_cumulative_visceral_slide

    Parameters
    ----------
    argv: list of str
        Command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help="a path to image folder in the cine-MRI archive")
    parser.add_argument('--masks', type=str, required=True, help="a path to segmentation masks of the whole slices")
    parser.add_argument('--slices_file', type=str, required=True, help="a path to a file with fill ids of slices to compute visceral slide for")
    parser.add_argument('--output', type=str, required=True, help="a path to a folder to save visceral slide")
    # Boolean flags
    parser.add_argument('--series_stat', dest='series_stat', action='store_true')
    parser.add_argument('--no-series_stat', dest='series_stat', action='store_false')
    parser.set_defaults(series_stat=False)
    parser.add_argument('--rest_def', dest='rest_def', action='store_true')
    parser.add_argument('--no-rest_def', dest='rest_def', action='store_false')
    parser.set_defaults(rest_def=False)

    args = parser.parse_args(argv)

    images_path = Path(args.images)
    masks_path = Path(args.masks)
    slices_file_path = Path(args.slices_file)
    output_path = Path(args.output)
    series_stat = args.series_stat
    print(series_stat)
    rest_def = args.rest_def
    print(rest_def)

    slices = slices_from_full_ids_file(slices_file_path)
    compute_cumulative_visceral_slide(images_path, masks_path, slices, output_path, series_stat, rest_def)


if __name__ == '__main__':
    # Very first argument determines action
    actions = {
        "extract_frames": extract_insp_exp,
        "compute": compute,
        "pipeline": pipeline,
        "cumulative_vs": cumulative_visceral_slide
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: registration ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])

