"""Train a Logistic Regression classifier with 5-fold CV, with features
from extract_features.py"""
from src.contour import Evaluation
from src.datasets import dev_dataset
from src.classification import (
    load_pixel_features,
    get_feature_array,
    get_normalizer,
)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import dump

# Parameters
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
evaluation = Evaluation.anterior_wall

# TODO will be replaced by loading json folds
series_ids = dataset.series_instance_uids
patient_ids = []
for series_id in series_ids:
    patient_ids.append(dataset[series_id]["PatientID"])


cv = GroupKFold()
predictions_dict = {}
model_idx = 0
for train_index, test_index in tqdm(
    cv.split(series_ids, groups=patient_ids), desc="Training classifiers"
):
    train_series, test_series = (
        np.array(series_ids)[train_index],
        np.array(series_ids)[test_index],
    )
    train_features, train_labels = get_feature_array(features, train_series)
    test_features, test_labels = get_feature_array(features, test_series)
    normalizer = get_normalizer(train_features)

    # Fit Logistic regression classifier
    clf = LogisticRegression().fit(train_features, train_labels)
    dump(clf, model_dir / f"model-{model_idx}.joblib")

    model_idx += 1
