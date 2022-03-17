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
predictions_path = Path(
    "/home/bram/data/registration_method/predictions/predictions.json"
)
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
evaluation = Evaluation.anterior_wall


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
        test_features_unnorm, test_labels = get_feature_array(features, [series_id])
        test_features = normalizer.transform(test_features_unnorm)
        prediction = clf.predict_proba(test_features)[:, 1]

        # Convert to connected predictions
        # TODO get x,y somehow nicer from features
        pred_boxes = get_boxes_from_raw(
            prediction, test_features_unnorm[:, 1], test_features_unnorm[:, 2]
        )

        # Get patient id and study id
        sample = dataset[str(series_id)]
        patient_id = sample["PatientID"]
        study_id = sample["StudyInstanceUID"]

        # Save in predictions_dict
        prediction_list = []
        for p, conf in pred_boxes:
            box = [p.origin_x, p.origin_y, p.width, p.height]
            conf = float(conf)

            # p.assign_type_from_mask(mask_np)
            # if p.type == AdhesionType.unset:
            #     box_type = "unset"
            # if p.type == AdhesionType.pelvis:
            #     box_type = "pelvis"
            # if p.type == AdhesionType.anteriorWall:
            #     box_type = "anterior"
            # if p.type == AdhesionType.inside:
            #     box_type = "inside"
            #
            box_type = "anterior"
            prediction_list.append((box, conf, box_type))
        if patient_id not in predictions_dict:
            predictions_dict[patient_id] = {}
        if study_id not in predictions_dict[patient_id]:
            predictions_dict[patient_id][study_id] = {}
        predictions_dict[patient_id][study_id][str(series_id)] = prediction_list

with open(predictions_path, "w") as file:
    json.dump(predictions_dict, file)
