"""Evaluate model predictions"""
from src.adhesions import AdhesionType, load_predictions
from src.evaluation import picai_eval
from pathlib import Path
from src.classification import get_boxes_from_raw
import matplotlib.pyplot as plt
import json

# Parameters
raw_predictions_path = Path(
    "/home/bram/data/registration_method/predictions_raw/raw_predictions.json"
)
predictions_path = Path(
    "/home/bram/data/registration_method/predictions/predictions.json"
)
extended_annotations_path = Path(
    "/home/bram/data/registration_method/extended_annotations.json"
)

# Load raw predictions
with open(raw_predictions_path) as json_file:
    raw_predictions = json.load(json_file)

# Convert to bounding box format
prediction_dict = {}
for patient_id in raw_predictions:
    for study_id in raw_predictions[patient_id]:
        for series_id in raw_predictions[patient_id][study_id]:
            prediction_list = []
            for region, pred_dict in raw_predictions[patient_id][study_id][
                series_id
            ].items():
                pred_boxes = get_boxes_from_raw(
                    pred_dict["prediction"], pred_dict["x"], pred_dict["y"]
                )

                for p, conf in pred_boxes:
                    box = [p.origin_x, p.origin_y, p.width, p.height]
                    conf = float(conf)
                    box_type = region
                    prediction_list.append((box, conf, box_type))

            if patient_id not in prediction_dict:
                prediction_dict[patient_id] = {}
            if study_id not in prediction_dict[patient_id]:
                prediction_dict[patient_id][study_id] = {}
            prediction_dict[patient_id][study_id][series_id] = prediction_list

        with open(predictions_path, "w") as file:
            json.dump(prediction_dict, file)

# Load predictions
predictions = load_predictions(predictions_path)

# Load annotations
annotations = load_predictions(extended_annotations_path)

metrics = picai_eval(
    predictions,
    annotations,
    flat=True,
    types=[AdhesionType.anteriorWall, AdhesionType.pelvis],
)

# Plot FROC
plt.figure()
plt.xlabel("Mean number of FPs per image")
plt.ylabel("Sensitivity")
plt.ylim([0, 1])
plt.xscale("log")
plt.plot(metrics["FP_per_case"], metrics["sensitivity"])
plt.show()

# Plot ROC
plt.figure()
plt.plot(metrics["fpr"], metrics["tpr"])
plt.show()
