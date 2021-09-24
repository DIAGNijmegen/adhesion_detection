#!/usr/local/bin/python3

import numpy as np
import json
import sys
import argparse
from pathlib import Path
import SimpleITK as sitk
from config import *
from utils import get_insp_exp_frames_and_masks, patients_from_full_ids_file
from cinemri.registration import Registrator
from cinemri.contour import mask_to_contour


# Extraction of the data necessary to compute visceral slide with different normalization options
# The necessary data is deformation fields and moving masks
# These data are saved separately for two groups of patients: training and control
# The folder's hierarchy is the following:
# vs_input
#    control
#    ...
#    train
#       insp_exp
#          moving_masks
#             patient_id
#                study_id
#                   slice_id.npy
#          df_rest
#          df_cavity
#          ...
#          df_complete
#             patient_id
#                study_id
#                   slice_id.npy
#       cumulative
#          moving_masks
#             patient_id
#                study_id
#                   slice_id
#                      0.npy
#                      ...
#                      29.npy
#          df_rest
#          df_cavity
#          df_contour
#          ...
#          df_complete
#             patient_id
#                study_id
#                   slice_id
#                      0.npy
#                      ...
#                      29.npy


def extract_insp_exp_dfs(patients, images_folder, segmentation_folder, insp_exp_path, output_folder):
    """
    Extracts moving masks and deformation fields for visceral slide calculation with inspiration and expiration frames
    Parameters
    ----------
    patients : list of Patient
       A list of patients to extract moving masks and deformation fields for
    images_folder : Path
       A path to cine-MRI archive
    segmentation_folder : Path
       A path to segmentations for cine-MRI slices in the archive
    insp_exp_path : Path
       A path to a file with inspiration/expiration frames positions information
    output_folder : Path
       A path where to save the extracted moving masks and deformation fields
    """

    # Create target folders
    insp_exp_folder = output_folder / INS_EXP_VS_FOLDER
    insp_exp_folder.mkdir()

    masks_folder = insp_exp_folder / MASKS_FOLDER
    masks_folder.mkdir()

    df_rest_folder = insp_exp_folder / DF_REST_FOLDER
    df_rest_folder.mkdir()

    df_cavity_folder = insp_exp_folder / DF_CAVITY_FOLDER
    df_cavity_folder.mkdir()

    df_complete_folder = insp_exp_folder / DF_COMPLETE_FOLDER
    df_complete_folder.mkdir()

    with open(insp_exp_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    registrator = Registrator()

    for patient in patients:
        patient_masks_folder = masks_folder / patient.id
        patient_masks_folder.mkdir()

        patient_df_rest_folder = df_rest_folder / patient.id
        patient_df_rest_folder.mkdir()

        patient_df_cavity_folder = df_cavity_folder / patient.id
        patient_df_cavity_folder.mkdir()

        patient_df_complete_folder = df_complete_folder / patient.id
        patient_df_complete_folder.mkdir()

        for study in patient.studies:
            study_masks_folder = patient_masks_folder / study.id
            study_masks_folder.mkdir()

            study_df_rest_folder = patient_df_rest_folder / study.id
            study_df_rest_folder.mkdir()

            study_df_cavity_folder = patient_df_cavity_folder / study.id
            study_df_cavity_folder.mkdir()

            study_df_complete_folder = patient_df_complete_folder / study.id
            study_df_complete_folder.mkdir()

            for slice in study.slices:
                insp_frame, insp_mask, exp_frame, exp_mask = get_insp_exp_frames_and_masks(slice,
                                                                                           inspexp_data,
                                                                                           images_folder,
                                                                                           segmentation_folder)

                # Save mask
                moving_mask_path = slice.build_path(masks_folder, extension=".npy")
                np.save(moving_mask_path, insp_mask)

                # complete DF
                _, complete_df = registrator.register(exp_frame, insp_frame)
                complete_df_path = slice.build_path(df_complete_folder, extension=".npy")
                np.save(complete_df_path, complete_df)

                # rest DF
                rest_df = registrator.get_masked_deformation_field(exp_frame, insp_frame, 1 - exp_mask, 1 - insp_mask)
                rest_df_path = slice.build_path(df_rest_folder, extension=".npy")
                np.save(rest_df_path, rest_df)

                # cavity DF
                cavity_df = registrator.get_masked_deformation_field(exp_frame, insp_frame, exp_mask, insp_mask)
                cavity_df_path = slice.build_path(df_cavity_folder, extension=".npy")
                np.save(cavity_df_path, cavity_df)


def extract_cumulative_dfs(patients, images_folder, segmentation_folder, output_folder):
    """
    Extracts moving masks and deformation fields for cumulative visceral slide calculation
    Parameters
    ----------
    patients : list of Patient
        A list of patients to extract moving masks and deformation fields for
    images_folder : Path
        A path to cine-MRI archive
    segmentation_folder : Path
        A path to segmentations for cine-MRI slices in the archive
    output_folder : Path
        A path where to save the extracted moving masks and deformation fields
    """
    
    # Create target folders
    cumulative_folder = output_folder / CUMULATIVE_VS_FOLDER
    cumulative_folder.mkdir()

    masks_folder = cumulative_folder / MASKS_FOLDER
    masks_folder.mkdir()

    df_rest_folder = cumulative_folder / DF_REST_FOLDER
    df_rest_folder.mkdir()

    df_cavity_folder = cumulative_folder / DF_CAVITY_FOLDER
    df_cavity_folder.mkdir()

    df_complete_folder = cumulative_folder / DF_COMPLETE_FOLDER
    df_complete_folder.mkdir()

    df_contour_folder = cumulative_folder / DF_CONTOUR_FOLDER
    df_contour_folder.mkdir()

    registrator = Registrator()

    for patient in patients:
        patient_masks_folder = masks_folder / patient.id
        patient_masks_folder.mkdir()

        patient_df_rest_folder = df_rest_folder / patient.id
        patient_df_rest_folder.mkdir()

        patient_df_cavity_folder = df_cavity_folder / patient.id
        patient_df_cavity_folder.mkdir()

        patient_df_complete_folder = df_complete_folder / patient.id
        patient_df_complete_folder.mkdir()

        patient_df_contour_folder = df_contour_folder / patient.id
        patient_df_contour_folder.mkdir()

        for study in patient.studies:
            study_masks_folder = patient_masks_folder / study.id
            study_masks_folder.mkdir()

            study_df_rest_folder = patient_df_rest_folder / study.id
            study_df_rest_folder.mkdir()

            study_df_cavity_folder = patient_df_cavity_folder / study.id
            study_df_cavity_folder.mkdir()

            study_df_complete_folder = patient_df_complete_folder / study.id
            study_df_complete_folder.mkdir()

            study_df_contour_folder = patient_df_contour_folder / study.id
            study_df_contour_folder.mkdir()

            for slice in study.slices:
                slice_masks_folder = study_masks_folder / slice.id
                slice_masks_folder.mkdir()

                slice_df_rest_folder = study_df_rest_folder / slice.id
                slice_df_rest_folder.mkdir()

                slice_df_cavity_folder = study_df_cavity_folder / slice.id
                slice_df_cavity_folder.mkdir()

                slice_df_complete_folder = study_df_complete_folder / slice.id
                slice_df_complete_folder.mkdir()

                slice_df_contour_folder = study_df_contour_folder / slice.id
                slice_df_contour_folder.mkdir()

                # Read slice and mask
                slice_path = slice.build_path(images_folder)
                slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

                mask_path = slice.build_path(segmentation_folder)
                mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

                for i in range(1, len(slice_array)):
                    # Taking previous frames as moving and the next one as fixed
                    moving = slice_array[i - 1].astype(np.uint32)
                    moving_mask = mask_array[i - 1].astype(np.uint32)

                    fixed = slice_array[i].astype(np.uint32)
                    fixed_mask = mask_array[i].astype(np.uint32)

                    # Save mask
                    moving_mask_path = slice.build_path(masks_folder, extension="") / "{}".format(i)
                    np.save(moving_mask_path, moving_mask)

                    # complete DF
                    _, complete_df = registrator.register(fixed, moving)
                    complete_df_path = slice.build_path(df_complete_folder, extension="") / "{}".format(i)
                    np.save(complete_df_path, complete_df)

                    # rest DF
                    rest_df = registrator.get_masked_deformation_field(fixed, moving, 1 - fixed_mask,
                                                              1 - moving_mask)
                    rest_df_path = slice.build_path(df_rest_folder, extension="") / "{}".format(i)
                    np.save(rest_df_path, rest_df)

                    # cavity DF
                    cavity_df = registrator.get_masked_deformation_field(fixed, moving, fixed_mask, moving_mask)
                    cavity_df_path = slice.build_path(df_cavity_folder, extension="") / "{}".format(i)
                    np.save(cavity_df_path, cavity_df)

                    # contour DF
                    contour_value = np.iinfo(np.uint16).max
                    fixed_contour = mask_to_contour(fixed_mask, contour_value)
                    moving_contour = mask_to_contour(moving_mask, contour_value)

                    _, contour_df = registrator.register(fixed_contour, moving_contour)
                    contour_df_path = slice.build_path(df_contour_folder, extension="") / "{}".format(i)
                    np.save(contour_df_path, contour_df)


def extract_vs_input(full_ids_file, images_folder, segmentation_folder, insp_exp_path, output_folder):
    """
    Extracts the data (moving masks and deformation fields) necessary for visceral slide calculation
    Parameters
    ----------
    full_ids_file: Path
       A path to a file with full ids of slices which to extract the data for
    images_folder: Path
       A path to an image folder
    segmentation_folder: Path
       A path to a folder with full segmentations
    insp_exp_path: Path
       A path to a file with expiration/inspiration data
    output_folder: Path
       A path to a folder where to save the results
    """

    output_folder.mkdir(exist_ok=True, parents=True)
    patients = patients_from_full_ids_file(full_ids_file)

    if insp_exp_path is not None:
        extract_insp_exp_dfs(patients, images_folder, segmentation_folder, insp_exp_path, output_folder)

    extract_cumulative_dfs(patients, images_folder, segmentation_folder, output_folder)


def extract(argv):
    """A command line wrapper of extract_vs_input

    Parameters
    ----------
    argv: list of str
        Command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("ids_file", type=str, help="a path to a file with full ids of slices which to extract the data for")
    parser.add_argument('--images', type=str, help="a path to an image folder")
    parser.add_argument("--masks", type=str, required=True,
                        help="a path to a folder with full segmentations")
    parser.add_argument("--insp_exp", type=str, help="a path to a file with expiration/inspiration data")
    parser.add_argument('--output', type=str, required=False, help="a path to an output folder")

    args = parser.parse_args(argv)
    full_ids_file_path = Path(args.ids_file)
    images_path = Path(args.images)
    masks_path = Path(args.masks)
    insp_exp_file_path = Path(args.insp_exp) if args.insp_exp else None
    output_path = Path(args.output)

    extract_vs_input(full_ids_file_path, images_path, masks_path, insp_exp_file_path, output_path)


if __name__ == '__main__':

    actions = {
        "extract": extract,
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: data_conversion ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])
