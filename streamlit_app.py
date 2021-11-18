"""Visualize inference results from inference.py with streamlit"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import json
from cinemri.datamodules import CineMRIDataModule
from cinemri.utils import plot_image
import numpy as np
import pickle
import json

data_dir = Path("/home/bram/dc/phd/registration_method/visualization/data")
inference_dir = Path("/home/bram/data/registration_method")


def get_dataset_with_boxes():
    datamodule = CineMRIDataModule(0, 0)
    datamodule.setup()

    return datamodule.train_dataset.dataset


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


def load_prediction(series_id):
    input_path = inference_dir / "predictions.json"
    with open(input_path, "r") as json_file:
        predictions = json.load(json_file)

    for patient in predictions:
        for study in predictions[patient]:
            for series in predictions[patient][study]:
                if series == series_id:
                    return predictions[patient][study][series]

    return []


def plot_vs(series_id, prediction, plot_boxes):
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
    plot_image(ax, image, boxes=boxes)

    # Load vs
    x, y, values = load_vs(series_id)

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
dataset = get_dataset_with_boxes()
series_ids = get_series_ids(dataset)

# Select a series id
series_idx = st.sidebar.number_input("Series index", 0, len(series_ids) - 1)
series_id = series_ids[series_idx]
st.write("Currently viewing series", series_id)

# Load predictions
prediction = load_prediction(series_id)

# Toggle bounding boxes
boxes_toggle = st.sidebar.checkbox("Display boxes", value=True)

# Layout
col1, col2 = st.beta_columns(2)
with col1:
    video = load_video(series_id)
    st.video(video)

with col2:
    st.pyplot(plot_vs(series_id, prediction, plot_boxes=boxes_toggle))
