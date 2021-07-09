
# Constants
SEPARATOR = "_"

# Paths
DETECTION_PATH = "/Users/emartynova/Documents/AIForHealth/Project/data/cinemri_mha_detection"

# Folders
# General
IMAGES_FOLDER = "images"
SEGMENTATION_FOLDER = "cavity_segmentations"
METADATA_FOLDER = "metadata"
# Data split
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
CONTROL_FOLDER = "control"
TRAIN_SEGM_FOLDER = "train_segm"
SEGM_FRAMES_FOLDER = "segmentation_subset"
DIAG_NNUNET_FOLDER = "diag_nnunet"
FULL_SEGMENTATION_FOLDER = "full_segmentation"
# Visceral slide input
MASKS_FOLDER = "moving_masks"
DF_REST_FOLDER = "df_rest"
DF_CAVITY_FOLDER = "df_cavity"
DF_COMPLETE_FOLDER = "df_complete"
DF_CONTOUR_FOLDER = "df_contour"

# Metadata files
BB_ANNOTATIONS_FILE = "annotations.json"
BB_ANNOTATIONS_EXPANDED_FILE = "annotations_expanded.json"
ANNOTATIONS_TYPE_FILE = "annotations_type.json"
OLD_REPORT_FILE_NAME = "rijnstate.json"
REPORT_FILE_NAME = "metadata.json"
PATIENTS_METADATA_FILE_NAME = "patients.json"
PATIENTS_METADATA_FILE_OLD_NAME = "patients_old.json"
PATIENTS_MAPPING_FILE_NAME = "patient_id_mapping.json"
INSPEXP_FILE_NAME = "inspexp.json"

# JSON keys
TRAIN_PATIENTS_KEY = "train_patients"
TEST_PATIENTS_KEY = "test_patients"
ADHESIONS_KEY = "adhesion"

# Visceral slide
VISCERAL_SLIDE_FILE = "visceral_slide.pkl"

# Validation split files
TRAIN_TEST_SPLIT_FILE_NAME = "segm_train_test_split.json"
NEGATIVE_PATIENTS_FILE_NAME = "negative_patients_detection.txt"
NEGATIVE_SLICES_FILE_NAME = "negative_slices_detection.txt"
NEGATIVE_SPLIT_FILE_NAME = "negative_split.json"
NEGATIVE_SLICES_TRAIN_NAME = "negative_slices_train.txt"
NEGATIVE_SLICES_TEST_NAME = "negative_slices_test.txt"
# IDs sampled automatically
NEGATIVE_SLICES_TRAIN_SEGM_NAME = "negative_slices_train_segm.txt"
# IDs manually filtered by position
NEGATIVE_SLICES_TRAIN_SEGM_MANUAL_NAME = "negative_slices_train_segm_manual.txt"
TEST_SLICES_NAME = "test_slices.txt"
TEST_POSNEG_SPLIT_NAME = "test_pos_neg.json"
NEGATIVE_SLICES_CONTROL_NAME = "negative_slices_control.txt"
BB_ANNOTATED_SLICES_FILE_NAME = "bb_annotations_full_ids.txt"
DETECTION_SLICES_FILE_NAME = "detection_train_full_ids.txt"
PATIENTS_SPLIT_FILE_NAME = "patients_split.json"

# K-fold split files
DETECTION_PATIENT_FOLD_FILE_NAME = "detection_patients_folds.json"
DETECTION_SLICE_FOLD_FILE_NAME = "detection_slices_folds.json"
SEGMENTATION_PATIENT_FOLD_FILE_NAME = "segmentation_patients_folds.json"
SEGMENTATION_SLICE_FOLD_FILE_NAME = "segmentation_slices_folds.json"
