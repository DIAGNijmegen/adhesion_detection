"""Evaluate model predictions"""
from cinemri.config import META_PATH
from src.datasets import frank_500_dataset
from src.adhesions import (
    AdhesionType,
    load_predictions,
    load_segmentation_predictions,
    load_segmentation_annotations,
)
from src.evaluation import picai_eval_segmentations
from pathlib import Path
from src.classification import get_boxes_from_raw, get_segmentation_from_raw
import matplotlib.pyplot as plt
import json
import pickle

# Parameters
raw_predictions_path = Path(
    "/home/bram/data/registration_method/predictions_raw/raw_predictions.json"
)
predictions_path = Path(
    "/home/bram/data/registration_method/predictions/predictions.pkl"
)
extended_annotations_path = (
    META_PATH / "bounding_boxes" / "first_frame_with_region.json"
)
dataset = frank_500_dataset()

# Load raw predictions
with open(raw_predictions_path) as json_file:
    raw_predictions = json.load(json_file)

# Get segmentation predictions for all series
prediction_dict = {}
for patient_id in raw_predictions:
    for study_id in raw_predictions[patient_id]:
        for series_id in raw_predictions[patient_id][study_id]:
            prediction_list = []
            sample = dataset[series_id]
            image_size = sample["numpy"][0].shape
            prediction_map = {}
            for region, pred_dict in raw_predictions[patient_id][study_id][
                series_id
            ].items():

                region_prediction_map = get_segmentation_from_raw(
                    pred_dict["prediction"], pred_dict["x"], pred_dict["y"], image_size
                )
                prediction_map[region] = region_prediction_map

                # pred_boxes = get_boxes_from_raw(
                #     pred_dict["prediction"], pred_dict["x"], pred_dict["y"]
                # )
                # for p, conf in pred_boxes:
                #     box = [p.origin_x, p.origin_y, p.width, p.height]
                #     conf = float(conf)
                #     box_type = region
                #     prediction_list.append((box, conf, box_type))

            if patient_id not in prediction_dict:
                prediction_dict[patient_id] = {}
            if study_id not in prediction_dict[patient_id]:
                prediction_dict[patient_id][study_id] = {}
            prediction_dict[patient_id][study_id][series_id] = prediction_map

        with open(predictions_path, "w+b") as file:
            pickle.dump(prediction_dict, file)

# Load predictions
# predictions = load_predictions(predictions_path)
predictions = load_segmentation_predictions(predictions_path)

# Load annotations
annotations = load_segmentation_annotations(dataset)


# Evaluation settings
evalation_regions = {}
evalation_regions["anterior"] = [AdhesionType.anteriorWall]
evalation_regions["pelvis"] = [AdhesionType.pelvis]
# evalation_regions["complete"] = [AdhesionType.anteriorWall, AdhesionType.pelvis]

fig = plt.figure()
ax_roc = fig.add_subplot(122)
ax_froc = fig.add_subplot(121)

for region, evaluation_region in evalation_regions.items():
    metrics = picai_eval_segmentations(
        predictions, annotations, flat=True, types=evaluation_region, iou_threshold=0.1
    )

    # Plot FROC
    ax_froc.set_xlabel("Mean number of FPs per image")
    ax_froc.set_ylabel("Sensitivity")
    ax_froc.set_ylim([0, 1])
    ax_froc.set_xscale("log")
    ax_froc.plot(metrics["FP_per_case"], metrics["sensitivity"], label=region)
    ax_froc.set_title("FROC")

    # Plot ROC
    ax_roc.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    ax_roc.plot(metrics["fpr"], metrics["tpr"], label=f"AUC: {metrics['auroc']:.2f}")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC")

ax_froc.legend()
ax_roc.legend()
plt.show()
