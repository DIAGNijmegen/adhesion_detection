"""Run inference on all CV folds"""
from src.contour import Evaluation
from src.datasets import dev_dataset, folds
from src.classification import (
    load_pixel_features,
    get_feature_array,
    load_model,
)
from sklearn.model_selection import GroupKFold
from pathlib import Path
import json
from tqdm import tqdm

# Parameters
raw_predictions_path = Path(
    "/home/bram/data/registration_method/predictions/raw_predictions.json"
)
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
evaluation = {"anterior": Evaluation.anterior_wall, "pelvis": Evaluation.pelvis}


cv = GroupKFold()
predictions_dict = {}
for region in evaluation:
    print(f"{region} region")
    for fold_idx in tqdm(range(5), desc="Classifier inference"):
        val_series = folds[str(fold_idx)]["val"]
        val_features, val_labels = get_feature_array(features, val_series)

        clf = load_model(model_dir / f"{region}-model-{fold_idx}.joblib")

        # Predict all pixels on validation set
        for series_id in val_series:
            test_features, test_labels = get_feature_array(
                features, [series_id], evaluation=evaluation
            )
            prediction = clf.predict(test_features)

            # Get x, y coordinates
            x_y, _ = get_feature_array(
                features,
                [series_id],
                evaluation=evaluation,
                included_features=["x", "y"],
            )

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
            if region not in predictions_dict[patient_id][study_id][str(series_id)]:
                predictions_dict[patient_id][study_id][str(series_id)][region] = {}

            predictions_dict[patient_id][study_id][str(series_id)][region][
                "prediction"
            ] = list(prediction)
            predictions_dict[patient_id][study_id][str(series_id)][region]["x"] = list(
                x_y[:, 0]
            )
            predictions_dict[patient_id][study_id][str(series_id)][region]["y"] = list(
                x_y[:, 1]
            )

with open(raw_predictions_path, "w") as file:
    json.dump(predictions_dict, file)
