"""Evaluate model predictions"""
from src.adhesions import AdhesionType, Adhesion, load_annotations, load_predictions
from src.evaluation import picai_eval
from pathlib import Path
import matplotlib.pyplot as plt

# Parameters
predictions_path = Path(
    "/home/bram/data/registration_method/predictions/predictions.json"
)
extended_annotations_path = Path(
    "/home/bram/data/registration_method/extended_annotations.json"
)

# Load predictions
predictions = load_predictions(predictions_path)

# Load annotations
annotations = load_predictions(extended_annotations_path)

metrics = picai_eval(
    predictions, annotations, flat=True, types=[AdhesionType.anteriorWall]
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
