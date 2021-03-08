# Auxilary script for local use only

conda activate cinemri_segmentation

python data_conversion.py extract_segmentation "../../data/cinemri_mha/rijnstate" "../../data/cinemri_mha/segmentation_subset"
