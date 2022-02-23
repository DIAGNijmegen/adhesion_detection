"""Visualize inference results from inference.py with streamlit"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import json
from cinemri.visualisation import plot_frame
from cinemri.config import ARCHIVE_PATH
import numpy as np
import pickle
from src.contour import filter_out_high_curvature
from src.datasets import dev_dataset
from src.adhesions import load_annotations, load_predictions
from src.detection_pipeline import (
    plot_FROCs,
    plot_ROCs,
    compute_slice_level_ROC,
    froc_wrapper,
)

data_dir = Path("/home/bram/dc/phd/registration_method/visualization/data")
inference_dir = Path("/home/bram/data/registration_method")
predictions_path = inference_dir / "predictions"
annotations_path = (
    ARCHIVE_PATH / "metadata" / "bounding_box_annotations_first_frame.json"
)


def get_series_ids(dataset):
    numbered_series_ids = {}
    for idx, series_id in enumerate(dataset.series_instance_uids):
        numbered_series_ids[idx] = series_id
    return numbered_series_ids


@st.cache
def load_video(series_id):
    with open(data_dir / series_id / "movie.mp4", "rb") as video_file:
        return video_file.read()


@st.cache
def load_vs(series_id):
    input_path = None
    for path in (inference_dir / "visceral_slide").rglob(series_id):
        input_path = path / "visceral_slide.pkl"

    if input_path is None:
        return 0, 0, 0

    with open(input_path, "r+b") as input_file:
        vs_dict = pickle.load(input_file)
        return vs_dict["x"], vs_dict["y"], vs_dict["slide"]


def load_prediction(predictions, series_id):
    prediction = []
    for full_id in predictions:
        _, _, series = full_id.split("_")
        if series == series_id:
            prediction = predictions[full_id]

    prediction = [
        ([p.origin_x, p.origin_y, p.width, p.height], float(conf))
        for p, conf in prediction
    ]

    return prediction


def load_all_predictions():
    predictions_dict = {}
    for path in predictions_path.glob("*"):
        predictions_name = path.with_suffix("").name
        predictions_dict[predictions_name] = load_predictions(path)

    return predictions_dict


def plot_vs(series_id, prediction, plot_boxes, filter_high_curvature):
    sample = dataset[series_id]

    # Assemble ground truth boxes
    boxes = []
    if plot_boxes:
        for box in sample["box"]:
            boxes.append({"box": box, "color": "green"})
    # Assemble prediction boxes
    for box in prediction:
        boxes.append({"box": box[0], "color": "red", "label": f"{box[1]:.2f}"})
    image = sample["numpy"][0]
    fig, ax = plt.subplots()
    plot_frame(ax, image, boxes=boxes)

    # Load vs
    x, y, values = load_vs(series_id)

    # Filter out high curvature
    if filter_high_curvature:
        contour_subset = filter_out_high_curvature(x, y, values)
        x = contour_subset[:, 0]
        y = contour_subset[:, 1]
        values = contour_subset[:, 2]

    # Plot visceral slide
    vs_scatter = ax.scatter(x, y, s=5, c=values, cmap="jet")
    cbar = fig.colorbar(mappable=vs_scatter)
    range = np.max(values) - np.min(values)
    min = np.min(values) + 0.05 * range
    max = np.max(values) - 0.05 * range
    cbar.set_ticks([min, max])
    cbar.set_ticklabels(["Low", "High"])
    return fig


# Get dataset and series
dataset = dev_dataset()
series_ids = get_series_ids(dataset)

# Load predictions and annotations
predictions_dict = load_all_predictions()
annotations = load_annotations(annotations_path, as_dict=True)

# Select a predictions set
prediction_names = list(predictions_dict.keys())
prediction_name = st.sidebar.selectbox("Predictions set", prediction_names)
predictions = predictions_dict[prediction_name]

# Select predictions for metric overview
prediction_subset = st.sidebar.multiselect(
    "Predictions for metric plots", prediction_names, default=prediction_names
)

# Select a series id
series_idx = st.sidebar.number_input("Series index", 0, len(series_ids) - 1)
series_id = series_ids[series_idx]
st.write("Currently viewing series", series_id)

# Get metrics
slice_rocs = []
frocs = []
metrics_legend = []
for name, p in predictions_dict.items():
    if name in prediction_subset:
        slice_roc = compute_slice_level_ROC(p, annotations)
        slice_rocs.append(slice_roc)
        froc = froc_wrapper(p, annotations)
        frocs.append((froc[0], froc[2]))
        metrics_legend.append(name)

# Load specific prediction
prediction = load_prediction(predictions, series_id)

# Toggle bounding boxes
boxes_toggle = st.sidebar.checkbox("Display boxes", value=True)

# Toggle filtering high curvature
curvature_toggle = st.sidebar.checkbox("Filter high curvature", value=True)

# Layout
col1, col2 = st.beta_columns(2)
with col1:
    video = load_video(series_id)
    st.video(video)

with col2:
    st.pyplot(
        plot_vs(
            series_id,
            prediction,
            plot_boxes=boxes_toggle,
            filter_high_curvature=curvature_toggle,
        )
    )
    st.pyplot(plot_ROCs(slice_rocs, legends=metrics_legend))
    st.pyplot(plot_FROCs(frocs, legends=metrics_legend))
