# Functions to perform the final data split for detection data set
import random
import json
import numpy as np
from collections import Counter
from pathlib import Path
from enum import Enum, unique
from utils import patients_from_metadata, patients_from_full_ids_file, slices_full_ids_from_patients, get_segm_patients_ids
from cinemri.definitions import Patient, CineMRIMotionType, CineMRISlice, CineMRISlicePos
from cinemri.utils import get_patients
from adhesions import AdhesionAnnotation, AdhesionType
from adhesions import load_annotations
from cinemri.config import ARCHIVE_PATH
from cinemri.definitions import AnatomicalPlane, Study
from config import *
from sklearn.model_selection import KFold

PARAMEDIAN_SEGMENTATION_IDS_POS = ["CM0003", "CM0063", "CM0170", "CM0193", "CM0203", "CM0211", "CM0214", "CM0302",
                                   "CM0400", "CM0062", "CM0082", "CM0173", "CM0200", "CM0208", "CM0212", "CM0294",
                                   "CM0345", "CM0443"]


@unique
class NegativePatientTypes(Enum):
    train = 0
    train_segm = 1
    train_no_segm = 2
    test = 3
    control = 4


@unique
class PatientTypes(Enum):
    train = 0
    train_positive = 1
    train_negative = 2
    test = 3
    test_positive = 4
    test_negative = 5
    control = 6


# Positive slices statistics: motion type and slice position
def positive_annotations_stat(annotations, patients, mapping):
    motion_type_freq = Counter()
    slice_pos_freq = Counter()

    for annotation in annotations:
        patient = [patient for patient in patients if mapping[patient.id] == annotation.patient_id][0]
        cinemri_slice = [slice for slice in patient.cinemri_slices if slice.id == annotation.slice_id][0]
        motion_type_freq[cinemri_slice.motion_type] += 1
        slice_pos_freq[cinemri_slice.position] += 1

    print("Motion type:")
    print("Full: {} of {}, {:.2f}%".format(motion_type_freq[CineMRIMotionType.full], len(annotations),
                                           100 * motion_type_freq[CineMRIMotionType.full]/len(annotations)))
    print("Squeeze pelvis: {} of {}, {:.2f}%".format(motion_type_freq[CineMRIMotionType.squeeze], len(annotations),
                                                     100 * motion_type_freq[CineMRIMotionType.squeeze]/len(annotations)))

    print("Slice position:")
    print("Left: {} of {}, {:.2f}%".format(slice_pos_freq[CineMRISlicePos.left], len(annotations),
                                           100 * slice_pos_freq[CineMRISlicePos.left] / len(annotations)))
    print("Right: {} of {}, {:.2f}%".format(slice_pos_freq[CineMRISlicePos.right], len(annotations),
                                            100 * slice_pos_freq[CineMRISlicePos.right] / len(annotations)))
    print("Paramedian: {} of {}, {:.2f}%".format(slice_pos_freq[CineMRISlicePos.left] + slice_pos_freq[CineMRISlicePos.right], len(annotations),
                                                 100 * (slice_pos_freq[CineMRISlicePos.left] + slice_pos_freq[CineMRISlicePos.right]) / len(annotations)))
    print("Midline: {} of {}, {:.2f}%".format(slice_pos_freq[CineMRISlicePos.middle], len(annotations),
                                              100 * slice_pos_freq[CineMRISlicePos.middle] / len(annotations)))


def positive_stat(annotations_file_path, patients_metadata_file, mapping_path):
    patients = patients_from_metadata(patients_metadata_file)
    patients = [patient for patient in patients if patient.id.startswith("ANON")]

    with open(mapping_path) as f:
        mapping = json.load(f)

    print("Statistics of annotation of all types:")
    annotations = load_annotations(annotations_file_path)
    positive_annotations_stat(annotations, patients, mapping)

    print("Statistics of annotation that intersect the abdominal cavity contour:")
    annotations = load_annotations(annotations_file_path,
                                   adhesion_types=[AdhesionType.anteriorWall, AdhesionType.abdominalCavityContour])
    positive_annotations_stat(annotations, patients, mapping)

    print("Statistics of annotation that intersect the anterior wall:")
    annotations = load_annotations(annotations_file_path,
                                   adhesion_types=[AdhesionType.anteriorWall])
    positive_annotations_stat(annotations, patients, mapping)


# Positive slices segmentation stat (how many segmented slices each patient have)
# ID with paramedian segmentation - extract manually
def segmentation_pos_stat(annotations_file_path, segmentation_path):
    print("Patients with paramedian segmentation {}".format(len(PARAMEDIAN_SEGMENTATION_IDS_POS)))
    
    annotations = load_annotations(annotations_file_path)
    patients_all_ids = np.unique([annotation.patient_id for annotation in annotations])
    print("{} patients with BB annotations in total".format(len(patients_all_ids)))

    annotations_contour = load_annotations(annotations_file_path,
                                           adhesion_types=[AdhesionType.anteriorWall,
                                                           AdhesionType.abdominalCavityContour])
    patients_contour_ids = np.unique([annotation.patient_id for annotation in annotations_contour])
    print("{} patients with BB annotations, who have annotations that intersect contour".format(len(patients_contour_ids)))

    patients_inside_only_ids = set(patients_all_ids).difference(patients_contour_ids)
    print("{} patients with BB annotations, who have annotations only inside".format(
        len(patients_inside_only_ids)))

    segm_patients = get_patients(segmentation_path)
    segm_patients_ids = [patient.id for patient in segm_patients]

    patients_slices = {}
    with_paramedian_nums = []
    without_paramedian_nums = []
    total_slices = 0
    for patient in segm_patients:
        total_slices += len(patient.cinemri_slices)
        patients_slices[patient.id] = len(patient.cinemri_slices)
        if patient.id in PARAMEDIAN_SEGMENTATION_IDS_POS:
            with_paramedian_nums.append(len(patient.cinemri_slices))
        else:
            without_paramedian_nums.append(len(patient.cinemri_slices))

    print("{} segmented slices in total".format(total_slices))

    # Min, max, mean number of slices for those with and without paramedian slices
    print("Segmentation statistics for those with paramedian slices")
    print("Max segm num {}".format(np.max(with_paramedian_nums)))
    print("Min segm num {}".format(np.min(with_paramedian_nums)))
    print("Mean segm num {}".format(np.mean(with_paramedian_nums)))

    print("Segmentation statistics for those without paramedian slices")
    print("Max segm num {}".format(np.max(without_paramedian_nums)))
    print("Min segm num {}".format(np.min(without_paramedian_nums)))
    print("Mean segm num {}".format(np.mean(without_paramedian_nums)))

    segm_patients_ann_ids = set(segm_patients_ids).intersection(patients_contour_ids)
    print("Patients with segmentation and BB annotaions, {}:".format(len(segm_patients_ann_ids)))
    print(segm_patients_ann_ids)

    segm_patients_no_ann_ids = set(segm_patients_ids).difference(patients_contour_ids)
    print("Patients with segmentation without BB annotaions, {}:".format(len(segm_patients_no_ann_ids)))
    print(segm_patients_no_ann_ids)

    patients_ann_paramdeian = set(segm_patients_ann_ids).intersection(PARAMEDIAN_SEGMENTATION_IDS_POS)
    print("Patients with paramedian segmentation and BB annotaions, {}:".format(len(patients_ann_paramdeian)))
    print(patients_ann_paramdeian)

    patients_no_ann_paramdeian = set(segm_patients_no_ann_ids).intersection(PARAMEDIAN_SEGMENTATION_IDS_POS)
    print("Patients with paramedian segmentation without BB annotaions, {}:".format(len(patients_no_ann_paramdeian)))
    print(patients_no_ann_paramdeian)
    print(patients_slices[list(patients_no_ann_paramdeian)[0]])

    # BB annotations slices number stat
    annotation_slice_counter = Counter()
    for annotation in annotations_contour:
        annotation_slice_counter[annotation.patient_id] += 1

    annotation_slice_num = list(annotation_slice_counter.items())
    annotation_slice_num_more_than_one = [item for item in annotation_slice_num if item[1] > 1]
    print("Patients with more than one BB annotations:")
    print(annotation_slice_num_more_than_one)


def full_ids_to_file(full_ids, output_file_path):
    
    with open(output_file_path, "w") as f:
        for full_id in full_ids:
            f.write(full_id + "\n")


# Extract full ids of slices with BB annotations that have adhesions intersecting contour
def bb_annotations_to_full_ids_file(bb_annotations_path, output_file_path, adhesion_types=[AdhesionType.anteriorWall,
                                                                                           AdhesionType.abdominalCavityContour]):

    annotations = load_annotations(bb_annotations_path,
                                   adhesion_types=adhesion_types)
    full_ids = [annotation.full_id for annotation in annotations]
    full_ids_to_file(full_ids, output_file_path)

    
def slice_vs_suitable(slice):
    """
    Checkes whether a slice is suitable for the algorithm to calculate the visceral slide
    To calculate the visceral slide only slices in sagittal plane should be used,
    midline slices and full abdominal motion are prefferable.
    Also we should check whether a slice have normal number of frames
    Returns
    -------
    suitable : bool
        A boolean flag indicaing whether a slice is suitable
    """

    return (slice.anatomical_plane == AnatomicalPlane.sagittal and
            slice.motion_type == CineMRIMotionType.full and
            slice.position == CineMRISlicePos.middle and
            slice.complete)


def slice_annotation_suitable(slice):
    """
    Checkes whether a slice is suitable for the algorithm to calculate the visceral slide
    To calculate the visceral slide only slices in sagittal plane should be used,
    midline slices and full abdominal motion are prefferable.
    Also we should check whether a slice have normal number of frames
    Returns
    -------
    suitable : bool
        A boolean flag indicaing whether a slice is suitable
    """

    return (slice.anatomical_plane == AnatomicalPlane.sagittal and
            slice.motion_type == CineMRIMotionType.full and
            slice.complete)


def mapped_old_ids(mapping_path):
    # Read json
    with open(mapping_path) as f:
        mapping = json.load(f)

    old_ids = []
    for key, value in mapping.items():
        if key.startswith("ANON"):
            old_ids.append(value)

    return old_ids


def get_controversial_patients_ids(report_path, negative=True, ids_to_exclude=[]):
    with open(report_path) as f:
        report_data = json.load(f)

    adhesion_attr_value = 0 if negative else 1
    controversial_patients = []
    for patient_id, studies in report_data.items():
        negative_studies_ids = []
        for study_id, report in studies.items():
            if report[ADHESIONS_KEY] == adhesion_attr_value and patient_id not in ids_to_exclude:
                negative_studies_ids.append(study_id)

        if len(negative_studies_ids) > 0 and len(negative_studies_ids) != len(studies):
            controversial_patients.append(patient_id)

    print("Controversial patients:")
    print(controversial_patients)

    return controversial_patients


def load_patients_and_studies(report_path, negative=False, ids_to_exclude=[]):
    with open(report_path) as f:
        report_data = json.load(f)

    adhesion_attr_value = 0 if negative else 1
    negative_patients = {}
    for patient_id, studies in report_data.items():
        negative_studies_ids = []
        for study_id, report in studies.items():
            if report[ADHESIONS_KEY] == adhesion_attr_value and patient_id not in ids_to_exclude:
                negative_studies_ids.append(study_id)

        if len(negative_studies_ids) > 0:
            negative_patients[patient_id] = negative_studies_ids

    return negative_patients


# Sampling negative patients
def get_suitable_patient_subset(report_path, patients_metadata_path, ids_to_exclude, suitability_function, negative=True, full_ids_file_path=None):

    # Subset of the report data for the specified category of patients
    patients_and_studies = load_patients_and_studies(report_path, negative=negative, ids_to_exclude=ids_to_exclude)
    patient_ids = patients_and_studies.keys()
    # Get full set of patients with all metadata
    patients = patients_from_metadata(patients_metadata_path)

    # Filter out patients that do not belong to the specified category
    patients = [patient for patient in patients if patient.id in patient_ids]
    patients_subset = []
    all_suitable_slices = []

    for patient in patients:
        # Patient with only suitable studies and slices
        patient_filtered = Patient(patient.id, patient.age, patient.sex)
        # Those studies which have at least one suitable slice
        for study in patient.studies:
            if study.id in patients_and_studies[patient.id]:
                suitable_slices = [slice for slice in study.slices if suitability_function(slice)]
                if len(suitable_slices) > 0:
                    all_suitable_slices += suitable_slices
                    # Study with only suitable slices
                    study_filtered = Study(study.id, study.patient_id, study.date)
                    for slice in suitable_slices:
                        study_filtered.add_slice(slice)
                    patient_filtered.add_study(study_filtered)

        if len(patient_filtered.studies) > 0:
            patients_subset.append(patient_filtered)

    outcome = "negative" if negative else "positive"
    print("Number of {} patients with suitable slices {}".format(outcome, len(patients_subset)))
    print("Total number of suitable {} slices {}".format(outcome, len(all_suitable_slices)))

    patient_slices = [(patient.id, len(patient.cinemri_slices)) for patient in patients_subset if len(patient.cinemri_slices) > 1]
    print("Patients with more than one suitable slice:")
    print(patient_slices)

    # Save full ids of suitable slices to file
    if full_ids_file_path is not None:
        full_ids = [slice.full_id for slice in all_suitable_slices]
        full_ids_to_file(full_ids, full_ids_file_path)

    return patients_subset


def negative_patients_split(patients, split_file=None, train_file=None, test_file=None, control_file=None):
    # Necessary to get the current split of patients
    patients.sort(key=lambda patient: patient.id, reverse=False)

    # shuffle patients
    np.random.shuffle(patients)

    patients_train = []
    # For balance with patients subset with BB annotations we take one patient with 5 slices
    # and 3 patients with 2 slices
    patient_five_slices = [patient for patient in patients if len(patient.cinemri_slices) == 5][0]
    patients_train.append(patient_five_slices)

    patients_two_slices = [patient for patient in patients if len(patient.cinemri_slices) == 2][:3]
    patients_train.extend(patients_two_slices)

    train_ids = [patient.id for patient in patients_train]
    indices = [index for index, patient in enumerate(patients) if patient.id in train_ids]

    # Remove taken patients
    patients = np.delete(patients, indices)

    # Add 46 more patients to the training set
    patients_train.extend(patients[:46])
    # Split training into with and without segmentation
    np.random.shuffle(patients_train)
    # Take first 10 as with segmentation
    patients_train_segm = patients_train[:10]
    patients_train_no_segm = patients_train[10:]

    if train_file is not None:
        train_slices_fill_ids = slices_full_ids_from_patients(patients_train)
        full_ids_to_file(train_slices_fill_ids, train_file)

    # Move 15 next to test set
    patients_test = patients[46:61]
    if test_file is not None:
        test_slices_fill_ids = slices_full_ids_from_patients(patients_test)
        full_ids_to_file(test_slices_fill_ids, test_file)

    # Take last 34 as control
    patients_control = patients[61:]
    if control_file is not None:
        control_slices_fill_ids = slices_full_ids_from_patients(patients_control)
        full_ids_to_file(control_slices_fill_ids, control_file)

    train_segm_ids = [patient.id for patient in patients_train_segm]
    train_no_segm_ids = [patient.id for patient in patients_train_no_segm]
    test_ids = [patient.id for patient in patients_test]
    control_ids = [patient.id for patient in patients_control]

    split_dict = {"train": {"segm": train_segm_ids, "no_segm": train_no_segm_ids},
                  "test": test_ids,
                  "control": control_ids}

    # Write a file with negative patients data split
    if split_file is not None:
        
        with open(split_file, "w") as f:
            json.dump(split_dict, f)

    return split_dict


# Sample negative patients with suitable slices, split into train (with segmentation subset), test, control,
# sample more slices for segmentation
def prepare_negative_patients(report_path, patients_metadata_path, metadata_path):

    full_ids_file = metadata_path / NEGATIVE_SLICES_FILE_NAME
    split_file = metadata_path / NEGATIVE_SPLIT_FILE_NAME
    train_file = metadata_path / NEGATIVE_SLICES_TRAIN_NAME
    train_segm_file = metadata_path / NEGATIVE_SLICES_TRAIN_SEGM_NAME
    test_file = metadata_path / NEGATIVE_SLICES_TEST_NAME
    control_file = metadata_path / NEGATIVE_SLICES_CONTROL_NAME

    old_ids = mapped_old_ids(mapping_path)
    controversial_patients_ids = get_controversial_patients_ids(report_path, old_ids)
    ids_to_exclude = old_ids + controversial_patients_ids

    negative_patients = get_suitable_patient_subset(report_path,
                                                    patients_metadata_path,
                                                    ids_to_exclude,
                                                    slice_vs_suitable,
                                                    full_ids_file_path=full_ids_file)

    negative_split = negative_patients_split(negative_patients,
                                             split_file,
                                             train_file,
                                             test_file,
                                             control_file)

    # Sample slices for additional segmentation (not only midline)
    sample_train_segm_subset(report_path, patients_metadata_file_path, negative_split, slice_annotation_suitable,
                             train_segm_file)

    return negative_split, split_file


def sample_test_subset(report_path, mapping_path, patients_metadata_path, json_split, suitability_function, metadata_path=None):

    # Negative subset
    all_patient_ids = [patient.id for patient in patients_from_metadata(patients_metadata_path)]
    test_ids = get_negative_patient_ids(NegativePatientTypes.test, json_split)
    ids_to_exclude = set(all_patient_ids).difference(test_ids)
    negative_subset = get_suitable_patient_subset(report_path, patients_metadata_path, ids_to_exclude,
                                                  suitability_function)
    # We want to sample the same number of positive patients as negative
    patients_num = len(negative_subset)

    # Positive subset
    old_ids = mapped_old_ids(mapping_path)
    controversial_positive_ids = get_controversial_patients_ids(report_path, negative=False, ids_to_exclude=old_ids)
    ids_to_exclude = old_ids + controversial_positive_ids
    positive_subset = get_suitable_patient_subset(report_path, patients_metadata_path, ids_to_exclude,
                                                  suitability_function, negative=False)

    # Select the subset of a necessary size
    np.random.shuffle(positive_subset)
    positive_subset = positive_subset[:patients_num]

    subset = negative_subset + positive_subset

    negative_ids = [patient.id for patient in negative_subset]
    positive_ids = [patient.id for patient in positive_subset]
    split_dict = {"negative": negative_ids, "positive": positive_ids}

    if metadata_path is not None:
        full_ids_file_path = metadata_path / TEST_SLICES_NAME
        slices_fill_ids = slices_full_ids_from_patients(subset)
        full_ids_to_file(slices_fill_ids, full_ids_file_path)

        pos_neg_split_file_path = metadata_path / TEST_POSNEG_SPLIT_NAME
        with open(pos_neg_split_file_path, "w") as f:
            json.dump(split_dict, f)

    return subset, split_dict

# Extracts all suitable slices for negative patients in the training set that were selected for additional segmentation
def sample_train_segm_subset(report_path, patients_metadata_path, json_split, suitability_function, full_ids_file_path=None):
    all_patient_ids = [patient.id for patient in patients_from_metadata(patients_metadata_path)]
    test_ids = get_negative_patient_ids(NegativePatientTypes.train_segm, json_split)
    ids_to_exclude = set(all_patient_ids).difference(test_ids)
    negative_train_segm_subset = get_suitable_patient_subset(report_path, patients_metadata_path, ids_to_exclude,
                                                             suitability_function)

    if full_ids_file_path is not None:
        slices_fill_ids = slices_full_ids_from_patients(negative_train_segm_subset)
        full_ids_to_file(slices_fill_ids, full_ids_file_path)

    return negative_train_segm_subset


def complete_patients_split(bb_slices_file, negative_split, testset_dict, output_file):
    positive_train = patients_from_full_ids_file(bb_slices_file)
    positive_train_ids = [patient.id for patient in positive_train]

    split_dict = {"train": {"positive": positive_train_ids,
                            "negative": negative_split["train"]["segm"] + negative_split["train"]["no_segm"]},
                  "test": testset_dict,
                  "control": negative_split["control"]}

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(split_dict, f)

    return split_dict


def get_negative_patient_ids(type, split_json):

    if type == NegativePatientTypes.train:
        patient_ids = split_json["train"]["segm"] + split_json["train"]["no_segm"]
    elif type == NegativePatientTypes.train_segm:
        patient_ids = split_json["train"]["segm"]
    elif type == NegativePatientTypes.train_no_segm:
        patient_ids = split_json["train"]["no_segm"]
    elif type == NegativePatientTypes.test:
        patient_ids = split_json["test"]
    else:
        patient_ids = split_json["control"]

    return patient_ids


def get_patient_ids(type, split_json):

    if type == PatientTypes.train:
        patient_ids = split_json["train"]["positive"] + split_json["train"]["negative"]
    elif type == PatientTypes.train_positive:
        patient_ids = split_json["train"]["positive"]
    elif type == PatientTypes.train_negative:
        patient_ids = split_json["train"]["negative"]
    elif type == PatientTypes.test:
        patient_ids = split_json["test"]["positive"] + split_json["test"]["negative"]
    elif type == PatientTypes.test_positive:
        patient_ids = split_json["test"]["positive"]
    elif type == PatientTypes.test_negative:
        patient_ids = split_json["test"]["negative"]
    else:
        patient_ids = split_json["control"]

    return patient_ids


# K-fold split
def kfold_by_group(groups, reversed, folds_num=5):
    kf = KFold(n_splits=folds_num)
    folds_train = []
    folds_val = []

    for group_ind, group in enumerate(groups):
        group = np.array(group)
        np.random.shuffle(group)

        for index, (train_index, val_index) in enumerate(kf.split(group)):
            fold_train_ids = group[train_index]
            fold_val_ids = group[val_index]

            if group_ind == 0:
                folds_train.append(fold_train_ids)
                folds_val.append(fold_val_ids)
            else:
                reversed_index = folds_num - index - 1 if reversed[group_ind] else index
                folds_train[reversed_index] = np.concatenate((folds_train[reversed_index], fold_train_ids))
                folds_val[reversed_index] = np.concatenate((folds_val[reversed_index], fold_val_ids))

    folds = []
    for train_ids, val_ids in zip(folds_train, folds_val):
        fold = {"train": train_ids.tolist(), "val": val_ids.tolist()}
        folds.append(fold)

    return folds


def load_folds(file_path):

    with open(file_path) as f:
        folds_json = json.load(f)

    return folds_json


def detection_folds(patient_split, negative_split, folds_num=5, output_file=None):

    neg_patients_segm = get_negative_patient_ids(NegativePatientTypes.train_segm, negative_split)
    neg_patients_no_segm = get_negative_patient_ids(NegativePatientTypes.train_no_segm, negative_split)

    pos_patients = get_patient_ids(PatientTypes.train_positive, patient_split)
    pos_patients_segm = list(set(pos_patients).intersection(PARAMEDIAN_SEGMENTATION_IDS_POS))
    pos_patients_no_segm = list(set(pos_patients).difference(pos_patients_segm))

    folds = kfold_by_group([neg_patients_segm, neg_patients_no_segm, pos_patients_segm, pos_patients_no_segm],
                           [False, False, False, True],
                           folds_num=folds_num)

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(folds, f)

    return folds


def segmentation_folds(detection_folds_file, segm_path, patients_split, output_file):
    detection_folds = load_folds(detection_folds_file)
    train_patients = get_patient_ids(PatientTypes.train, patients_split)
    segm_patients_ids = get_segm_patients_ids(segm_path)

    additional_segm_ids = list(set(segm_patients_ids).difference(train_patients))
    additional_folds = kfold_by_group([additional_segm_ids], [False], len(detection_folds))

    segmentation_folds = []
    for detection_fold, additional_fold in zip(detection_folds, additional_folds):
        segmentation_train_subset = list(set(detection_fold["train"]).intersection(segm_patients_ids))
        segmentation_val_subset = list(set(detection_fold["val"]).intersection(segm_patients_ids))
        segmentation_fold_train = np.concatenate((segmentation_train_subset, additional_fold["train"]))
        segmentation_fold_val = np.concatenate((segmentation_val_subset, additional_fold["val"]))
        segmentation_folds.append({"train": segmentation_fold_train.tolist(), "val": segmentation_fold_val.tolist()})

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(segmentation_folds, f)
    
    return segmentation_folds


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(99)

    archive_path = ARCHIVE_PATH
    detection_path = Path(DETECTION_PATH)
    # Folders
    images_path = archive_path / IMAGES_FOLDER
    segmentation_path = detection_path / SEGMENTATION_FOLDER
    metadata_path = archive_path / METADATA_FOLDER
    annotation_path = metadata_path / ANNOTATIONS_TYPE_FILE

    # Files
    annotation_expanded_path = metadata_path / BB_ANNOTATIONS_EXPANDED_FILE
    patients_metadata_file_path = metadata_path / PATIENTS_METADATA_FILE_NAME
    mapping_path = metadata_path / PATIENTS_MAPPING_FILE_NAME
    report_path = metadata_path / REPORT_FILE_NAME
    bb_annotations_full_ids_file = metadata_path / BB_ANNOTATED_SLICES_FILE_NAME

    # Data split files

    test_slices_file = metadata_path / TEST_SLICES_NAME
    patient_split_file = metadata_path / PATIENTS_SPLIT_FILE_NAME
    # K-fold
    detection_kfold_file = metadata_path / DETECTION_PATIENT_FOLD_FILE_NAME
    segmentation_kfold_file = metadata_path / SEGMENTATION_PATIENT_FOLD_FILE_NAME

    # Auxiliary data
    # Patients ids to exclude from sampling of negative patients: old data set and patients with not positive 
    # and negative studies

    # BB annotations statistics
    # positive_stat(annotation_expanded_path, patients_metadata_old_file_path, mapping_path)
    # segmentation_pos_stat(annotation_expanded_path, segmentation_path)

    # BB annotations with suitable slices to file of full ids of slices (58 slices, 50 patients) 
    # bb_annotations_to_full_ids_file(annotation_expanded_path, bb_annotations_full_ids_file)
    
    # Negative patients with suitable slices to file of full ids of slices
    negative_split, negative_split_file = prepare_negative_patients(report_path, patients_metadata_file_path, metadata_path)

    # Sample test subset
    _, testset_dict = sample_test_subset(report_path, mapping_path, patients_metadata_file_path, negative_split, slice_annotation_suitable, metadata_path=metadata_path)

    # Make data split by patient
    patient_split = complete_patients_split(bb_annotations_full_ids_file, negative_split, testset_dict, patient_split_file)

    # Detection folds
    dfolds = detection_folds(patient_split, negative_split, output_file=detection_kfold_file)
    
    # Segmentation folds
    sfolds = segmentation_folds(detection_kfold_file, segmentation_path, patient_split, segmentation_kfold_file)

