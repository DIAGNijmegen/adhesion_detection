# Auxilary script for local use only

conda activate cinemri_segmentation

python data_conversion.py to_diag_nnunet "../../data/cinemri_mha/segmentation_subset" "../../data/cinemri_mha/diag_nnunet"