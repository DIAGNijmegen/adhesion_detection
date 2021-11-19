"""Development and test datasets"""
import json
import csv
from cinemri.config import ARCHIVE_PATH
from cinemri.dataset import CineMRIDataset

folds_path = "/home/bram/repos/abdomenmrus-cinemri/data/folds_20211015.json"
test_split_path = ARCHIVE_PATH / "metadata" / "registration_test_set.csv"

# Get folds from json
with open(folds_path) as json_file:
    folds = json.load(json_file)

# Get list of all bounding box annotated series
fold = folds[str(0)][str(0)]
all_series_ids = fold["train"] + fold["val"] + fold["test"]

# Load registration test set list
test_series_ids = []
with open(test_split_path, "r") as thefile:
    csvFile = csv.reader(thefile)

    for lines in csvFile:
        test_series_ids.append(lines[0])

# Get mutually exclusive series id list
dev_series_ids = []
for series_id in all_series_ids:
    if series_id in test_series_ids:
        continue
    dev_series_ids.append(series_id)


def dev_dataset():
    # Get general archive wrapper dataset
    dataset = CineMRIDataset(
        ARCHIVE_PATH, load_heatmaps=False, load_study_labels=False, cache=False
    )
    # Remove series not in dev list
    for series_instance_uid in dataset.series_instance_uids.copy():
        if series_instance_uid not in dev_series_ids:
            dataset._remove_image(series_instance_uid)

    return dataset
