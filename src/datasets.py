"""Development and test datasets"""
import json
from cinemri.config import ARCHIVE_PATH
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
    dataset = CineMRIDataset(
        ARCHIVE_PATH, load_heatmaps=False, load_study_labels=False, cache=False
    )
    # Remove series not in dev list
    for series_instance_uid in dataset.series_instance_uids.copy():
        if series_instance_uid not in dev_series_ids:
            dataset._remove_image(series_instance_uid)

    return dataset
