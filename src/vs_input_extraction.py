import numpy as np
import json
from pathlib import Path
import SimpleITK as sitk
from cinemri.registration import Registrator
from cinemri.utils import get_patients
from cinemri.contour import mask_to_contour
from utils import get_insp_exp_frames_and_masks, patients_from_full_ids_file
from config import *

VS_DATA_FOLDER_NAME = "vs_input"
CONTROL_GROUP_FOLDER_NAME = "control"
TRAIN_GROUP_FOLDER_NAME = "train"
INSP_EXP_FOLDER_NAME = "insp_exp"
CUMULATIVE_FOLDER_NAME = "cumulative"
MASKS_FOLDER_NAME = "moving_masks"
DF_REST_FOLDER_NAME = "df_rest"
DF_CAVITY_FOLDER_NAME = "df_cavity"
DF_COMPLETE_FOLDER_NAME = "df_complete"
DF_CONTOUR_FOLDER_NAME = "df_contour"

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

    # Create target folders
    insp_exp_folder = output_folder / INSP_EXP_FOLDER_NAME
    insp_exp_folder.mkdir()

    masks_folder = insp_exp_folder / MASKS_FOLDER_NAME
    masks_folder.mkdir()

    df_rest_folder = insp_exp_folder / DF_REST_FOLDER_NAME
    df_rest_folder.mkdir()

    df_cavity_folder = insp_exp_folder / DF_CAVITY_FOLDER_NAME
    df_cavity_folder.mkdir()

    df_complete_folder = insp_exp_folder / DF_COMPLETE_FOLDER_NAME
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
                _, rest_df = registrator.get_masked_deformation_field(exp_frame, insp_frame, 1 - exp_mask, 1 - insp_mask)
                rest_df_path = slice.build_path(df_rest_folder, extension=".npy")
                np.save(rest_df_path, rest_df)

                # cavity DF
                _, cavity_df = registrator.get_masked_deformation_field(exp_frame, insp_frame, exp_mask, insp_mask)
                cavity_df_path = slice.build_path(df_cavity_folder, extension=".npy")
                np.save(cavity_df_path, cavity_df)


def extract_cumulative_dfs(patients, images_folder, segmentation_folder, output_folder):

    # Create target folders
    cumulative_folder = output_folder / CUMULATIVE_FOLDER_NAME
    cumulative_folder.mkdir()

    masks_folder = cumulative_folder / MASKS_FOLDER_NAME
    masks_folder.mkdir()

    df_rest_folder = cumulative_folder / DF_REST_FOLDER_NAME
    df_rest_folder.mkdir()

    df_cavity_folder = cumulative_folder / DF_CAVITY_FOLDER_NAME
    df_cavity_folder.mkdir()

    df_complete_folder = cumulative_folder / DF_COMPLETE_FOLDER_NAME
    df_complete_folder.mkdir()

    df_contour_folder = cumulative_folder / DF_CONTOUR_FOLDER_NAME
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
                    _, rest_df = registrator.get_masked_deformation_field(fixed, moving, 1 - fixed_mask,
                                                              1 - moving_mask)
                    rest_df_path = slice.build_path(df_rest_folder, extension="") / "{}".format(i)
                    np.save(rest_df_path, rest_df)

                    # cavity DF
                    _, cavity_df = registrator.get_masked_deformation_field(fixed, moving, fixed_mask, moving_mask)
                    cavity_df_path = slice.build_path(df_cavity_folder, extension="") / "{}".format(i)
                    np.save(cavity_df_path, cavity_df)

                    # contour DF
                    contour_value = np.iinfo(np.uint16).max
                    fixed_contour = mask_to_contour(fixed_mask, contour_value)
                    moving_contour = mask_to_contour(moving_mask, contour_value)

                    _, contour_df = registrator.register(fixed_contour, moving_contour)
                    contour_df_path = slice.build_path(df_contour_folder, extension="") / "{}".format(i)
                    np.save(contour_df_path, contour_df)


def extract_vs_input(images_folder, full_ids_file, segmentation_folder, insp_exp_path, output_folder):

    output_folder.mkdir(exist_ok=True, parents=True)
    
    patients = patients_from_full_ids_file(full_ids_file)

    extract_insp_exp_dfs(patients, images_folder, segmentation_folder, insp_exp_path, output_folder)
    extract_cumulative_dfs(patients, images_folder, segmentation_folder, output_folder)


def extract_vs_input_train_control(train_images_folder,
                                   train_full_ids_file,
                                   control_images_folder,
                                   control_full_ids_file,
                                   segmentation_folder,
                                   insp_exp_path,
                                   output_folder):

    vs_input_folder = output_folder / VS_DATA_FOLDER_NAME
    vs_input_folder.mkdir(exist_ok=True)

    train_group_folder = vs_input_folder / TRAIN_GROUP_FOLDER_NAME
    extract_vs_input(train_images_folder, train_full_ids_file, segmentation_folder, insp_exp_path, train_group_folder)

    #control_group_folder = vs_input_folder / CONTROL_GROUP_FOLDER_NAME
    #extract_vs_input(control_images_folder, control_full_ids_file, segmentation_folder, insp_exp_path, control_group_folder)


if __name__ == '__main__':

    detection_path = Path(DETECTION_PATH)
    train_images_path = detection_path / IMAGES_FOLDER / TRAIN_FOLDER
    train_images_ids_path = detection_path / METADATA_FOLDER / "detection_train_full_ids1.txt"
    control_images_path = detection_path / IMAGES_FOLDER / CONTROL_FOLDER
    control_images_ids_path = detection_path / METADATA_FOLDER / "negative_slices_control1.txt"
    full_segm_path = detection_path / FULL_SEGMENTATION_FOLDER
    insp_exp_path = detection_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    output_folder = detection_path / "output2"

    extract_vs_input_train_control(train_images_path,
                                   train_images_ids_path,
                                   control_images_path,
                                   control_images_ids_path,
                                   full_segm_path,
                                   insp_exp_path,
                                   output_folder)
