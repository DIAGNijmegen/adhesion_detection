"""Run inference on all CV folds"""
from src.contour import Evaluation
from src.datasets import dev_dataset, folds
from src.classification import (
    load_pixel_features,
    get_feature_array,
    get_normalizer,
    get_boxes_from_raw,
)
from sklearn.model_selection import GroupKFold
from pathlib import Path
import json
from joblib import load
from tqdm import tqdm

# Parameters
raw_predictions_path = Path(
    "/home/bram/data/registration_method/predictions/raw_predictions.json"
)
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
evaluation = Evaluation.pelvis


cv = GroupKFold()
predictions_dict = {}
for fold_idx in tqdm(range(5), desc="Classifier inference"):
    train_series = folds[str(fold_idx)]["train"]
    train_features, train_labels = get_feature_array(features, train_series)
    val_series = folds[str(fold_idx)]["val"]
    val_features, val_labels = get_feature_array(features, val_series)
    normalizer = get_normalizer(train_features)

    # Load model
    clf = load(model_dir / f"model-{fold_idx}.joblib")

    # Predict all pixels on validation set
    for series_id in val_series:
        test_features, test_labels = get_feature_array(
            features, [series_id], evaluation=evaluation
        )
        test_features = normalizer.transform(test_features)
        prediction = clf.predict_proba(test_features)[:, 1]

        # Get x, y coordinates
        x_y, _ = get_feature_array(
            features, [series_id], evaluation=evaluation, included_features=["x", "y"]
        )

        # Convert to connected predictions
        pred_boxes = get_boxes_from_raw(prediction, x_y[:, 0], x_y[:, 1])

        # Get patient id and study id
        sample = dataset[str(series_id)]
        patient_id = sample["PatientID"]
        study_id = sample["StudyInstanceUID"]

        if patient_id not in predictions_dict:
            predictions_dict[patient_id] = {}
        if study_id not in predictions_dict[patient_id]:
            predictions_dict[patient_id][study_id] = {}
        if series_id not in predictions_dict[patient_id][study_id]:
            predictions_dict[patient_id][study_id][str(series_id)] = {}

        predictions_dict[patient_id][study_id][str(series_id)]["prediction"] = list(
            prediction
        )
        predictions_dict[patient_id][study_id][str(series_id)]["x"] = list(x_y[:, 0])
        predictions_dict[patient_id][study_id][str(series_id)]["y"] = list(x_y[:, 1])

with open(raw_predictions_path, "w") as file:
    json.dump(predictions_dict, file)
