
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
# Detection algorithm input
VS_FOLDER = "visceral_slide"
VS_CONTROL_FOLDER = "visceral_slide_control"
VS_TEST_FOLDER = "visceral_slide_test"
UNNORM_FOLDER = "unnorm"
AVG_NORM_FOLDER = "avg_norm"
VICINITY_NORM_FOLDER = "vicinity_norm"
CUMULATIVE_VS_FOLDER = "cumulative"
INS_EXP_VS_FOLDER = "insp_exp"

# Metadata files
BB_ANNOTATIONS_FILE = "annotations.json"
BB_ANNOTATIONS_EXPANDED_FILE = "annotations_expanded.json"
BB_TEST_ANNOTATIONS_FILE = "annotations_test.json"
BB_TEST_ANNOTATIONS_EXPANDED_FILE = "annotations_test_expanded.json"
ANNOTATIONS_TYPE_FILE = "annotations_type.json"
OLD_REPORT_FILE_NAME = "rijnstate.json"
REPORT_FILE_NAME = "metadata.json"
PATIENTS_METADATA_FILE_NAME = "patients.json"
PATIENTS_METADATA_FILE_OLD_NAME = "patients_old.json"
PATIENTS_MAPPING_FILE_NAME = "patient_id_mapping.json"
INSPEXP_FILE_NAME = "inspexp.json"

# Output files
EVALUATION_METRICS_FILE = "metrics.pkl"

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
# Visceral slide expectation abdominal motion norm
CUMULATIVE_VS_EXPECTATION_FILE = "cumulative_vs_expectation.pkl"
INSPEXP_VS_EXPECTATION_FILE = "insexp_vs_expectation.pkl"
# SQRT transformed abdominal motion norm
CUMULATIVE_VS_EXPECTATION_FILE_SQRT = "cumulative_vs_expectation_sqrt.pkl"
INSPEXP_VS_EXPECTATION_FILE_SQRT = "insexp_vs_expectation_sqrt.pkl"
# Visceral slide expectation vicinity norm
CUMULATIVE_VS_EXPECTATION_VICINITY_FILE = "cumulative_vs_expectation_vicinity.pkl"
INSPEXP_VS_EXPECTATION_VICINITY_FILE = "inspexp_vs_expectation_vicinity.pkl"
# SQRT transformed vicinity norm
CUMULATIVE_VS_EXPECTATION_VICINITY_FILE_SQRT = "cumulative_vs_expectation_vicinity_sqrt.pkl"
INSPEXP_VS_EXPECTATION_VICINITY_FILE_SQRT = "inspexp_vs_expectation_vicinity_sqrt.pkl"

# K-fold split files
DETECTION_PATIENT_FOLD_FILE_NAME = "detection_patients_folds.json"
DETECTION_SLICE_FOLD_FILE_NAME = "detection_slices_folds.json"
SEGMENTATION_PATIENT_FOLD_FILE_NAME = "segmentation_patients_folds.json"
SEGMENTATION_SLICE_FOLD_FILE_NAME = "segmentation_slices_folds.json"
