"""Development and test datasets"""
import json
from cinemri.config import ARCHIVE_PATH, META_PATH
from cinemri.dataset import CineMRIDataset

folds_path = "/home/bram/repos/abdomenmrus-cinemri/data/folds_20211015.json"
folds_path = "/home/bram/repos/abdomenmrus-cinemri-patch-level/folds_20220317.json"
test_split_path = ARCHIVE_PATH / "metadata" / "registration_test_set.csv"

# Get folds from json
with open(folds_path) as json_file:
    folds = json.load(json_file)

# Load registration test set list
test_series_ids = folds["test"]

# Get mutually exclusive series id list
dev_series_ids = []
fold = folds[str(0)]
dev_series_ids = fold["train"] + fold["val"]


def dev_dataset():
    """Dataset containing all annotated series, excluding those that are
    in the 90-slice test set annotated by Frank"""
    # Get general archive wrapper dataset
    dataset = CineMRIDataset()
    # Remove series not in dev list
    for series_instance_uid in dataset.series_instance_uids.copy():
        if series_instance_uid not in dev_series_ids:
            dataset._remove_image(series_instance_uid)

    return dataset


def test_dataset():
    """Dataset containing all annotated series that are
    in the 90-slice test set annotated by Frank"""
    # Get general archive wrapper dataset
    dataset = CineMRIDataset()
    # Remove series not in dev list
    for series_instance_uid in dataset.series_instance_uids.copy():
        if series_instance_uid not in test_series_ids:
            dataset._remove_image(series_instance_uid)

    return dataset


# New dataset (frank_500), excluding scans that were used for nnu-net training
folds_path = META_PATH / "folds" / "nested_5_fold_20220422.json"

# Get series_ids from folds
with open(folds_path) as json_file:
    folds = json.load(json_file)
fold = folds["0"]["0"]
series_ids_frank_500 = fold["train"] + fold["val"] + fold["test"]

# Get series ids with cavity segmentation
cavity_path = META_PATH / "cavity_segmentations"
series_ids_cavity = []
for series_id in cavity_path.rglob("*.npy"):
    # Remove .npy
    series_id = series_id.name[:-4]

    # Add .0
    series_id += ".0"

    series_ids_cavity.append(series_id)

final_series_ids = []
for series_id in series_ids_frank_500:
    if series_id in series_ids_cavity:
        continue

    final_series_ids.append(series_id)


def frank_500_dataset():
    """Dataset containing all annotated series the new frank_500 set,
    excluding series that were used for nnunet training"""
    # Get general archive wrapper dataset
    dataset = CineMRIDataset(load_adhesion_segmentations=True)
    # Remove series not in dev list
    for series_instance_uid in dataset.series_instance_uids.copy():
        if series_instance_uid not in final_series_ids:
            dataset._remove_image(series_instance_uid)

    return dataset


frank_500_folds_path = "/home/bram/repos/adhesion_detection/src/folds_20220426.json"

# Get folds from json
with open(frank_500_folds_path) as json_file:
    frank_500_folds = json.load(json_file)
