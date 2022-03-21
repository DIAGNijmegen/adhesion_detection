"""Visualize inference results from inference.py with streamlit"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import json
from cinemri.visualisation import plot_frame, get_interactive_plotly_movie
from cinemri.config import ARCHIVE_PATH
import numpy as np
import pickle
from src.contour import (
    filter_out_high_curvature,
    filter_out_prior_vs_subset,
    Evaluation,
)
from src.datasets import dev_dataset
from src.adhesions import load_annotations, load_predictions, AdhesionType
from src.detection_pipeline import (
    plot_FROCs,
    plot_ROCs,
    compute_slice_level_ROC,
    froc_wrapper,
)
from src.vs_computation import calculate_motion_map
from src.evaluation import picai_eval

st.set_page_config(layout="wide")

inference_dir = Path("/home/bram/data/registration_method")
predictions_path = inference_dir / "predictions"
annotations_path = Path("/home/bram/data/registration_method/extended_annotations.json")
raw_predictions_path = Path(
    "/home/bram/data/registration_method/predictions_raw/raw_predictions.json"
)
with open(raw_predictions_path) as json_file:
    raw_predictions = json.load(json_file)


def get_series_ids(dataset):
    numbered_series_ids = {}
    for idx, series_id in enumerate(dataset.series_instance_uids):
        numbered_series_ids[idx] = series_id
    return numbered_series_ids


def load_motion_map(sample):
    visceral_slide_dir = Path("/home/bram/data/registration_method/visceral_slide")
    vs_computation_input_path = (
        visceral_slide_dir
        / sample["PatientID"]
        / sample["StudyInstanceUID"]
        / sample["SeriesInstanceUID"]
        / "vs_computation_input.pkl"
    )
    with open(vs_computation_input_path, "r+b") as pkl_file:
        vs_computation_input = pickle.load(pkl_file)

    return calculate_motion_map(vs_computation_input["normalization_dfs"])


def normalize_vs_by_motion(x, y, vs, motion_map):
    for i in range(len(x)):
        motion = motion_map[int(y[i]), int(x[i])]
        vs[i] = vs[i] / motion
    return vs


# @st.cache
def load_vs(input_series_id, adhesion_types):
    for patient_id in raw_predictions:
        for study_id in raw_predictions[patient_id]:
            for series_id, trial_pred_dict in raw_predictions[patient_id][
                study_id
            ].items():
                if series_id == input_series_id:
                    pred_dict = trial_pred_dict

    types = []
    for adhesion_type in adhesion_types:
        if adhesion_type == AdhesionType.anteriorWall:
            types.append("anterior")
        if adhesion_type == AdhesionType.pelvis:
            types.append("pelvis")

    x = []
    y = []
    values = []
    for region in types:
        x += pred_dict[region]["x"]
        y += pred_dict[region]["y"]
        values += pred_dict[region]["prediction"]

    return (
        x,
        y,
        values,
    )


def load_prediction(predictions, series_id):
    prediction = []
    for full_id in predictions:
        _, _, series = full_id.split("_")
        if series == series_id:
            prediction = predictions[full_id]

    prediction_list = []
    for entry in prediction:
        p = entry[0]
        conf = entry[1]
        box = [p.origin_x, p.origin_y, p.width, p.height]
        conf = float(conf)
        type = p.type

        prediction_list.append((box, conf, type))

    return prediction_list


def load_all_predictions():
    predictions_dict = {}
    for path in predictions_path.glob("*"):
        predictions_name = path.with_suffix("").name
        predictions_dict[predictions_name] = load_predictions(path)

    return predictions_dict


def plot_vs(
    series_id,
    prediction,
    annotation,
    adhesion_types,
    plot_label_boxes,
    plot_predicted_boxes,
    # filter_prior,
    # filter_high_curvature,
    # normalize_by_motion,
    # motion_map,
):
    sample = dataset[series_id]

    # Assemble ground truth boxes
    boxes = []
    if plot_label_boxes:
        for box in annotation:
            if box[2] not in adhesion_types:
                continue
            boxes.append({"box": box[0], "color": "green"})
    # Assemble prediction boxes
    if plot_predicted_boxes:
        for box in prediction:
            if box[2] not in adhesion_types:
                continue
            color = "red"
            boxes.append({"box": box[0], "color": color, "label": f"{box[1]:.2f}"})
    image = sample["numpy"][0]
    fig, ax = plt.subplots()
    plot_frame(ax, image, boxes=boxes)

    # Load raw prediction
    x, y, values = load_vs(series_id, adhesion_types)

    # Filter out prior
    # if filter_prior:
    #     contour_subset = filter_out_prior_vs_subset(
    #         x, y, values, evaluation=Evaluation.anterior_wall
    #     )
    #     x = contour_subset[:, 0]
    #     y = contour_subset[:, 1]
    #     values = contour_subset[:, 2]

    # Filter out high curvature
    # if filter_high_curvature:
    #     contour_subset = filter_out_high_curvature(x, y, values)
    #     x = contour_subset[:, 0]
    #     y = contour_subset[:, 1]
    #     values = contour_subset[:, 2]

    # if normalize_by_motion:
    #     values = normalize_vs_by_motion(x, y, values, motion_map)

    # Plot visceral slide
    vs_scatter = ax.scatter(x, y, s=5, vmin=0, vmax=1, c=values, cmap="jet")
    cbar = fig.colorbar(mappable=vs_scatter)
    # range = np.max(values) - np.min(values)
    # min = np.min(values) + 0.05 * range
    # max = np.max(values) - 0.05 * range
    cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(["Low", "High"])
    return fig


def plot_vs_flat(
    series_id,
    filter_prior,
    filter_high_curvature,
    normalize_by_motion,
    motion_map,
):
    fig, ax = plt.subplots()

    # Load vs
    x, y, values = load_vs(series_id)

    # Filter out prior
    if filter_prior:
        contour_subset = filter_out_prior_vs_subset(
            x, y, values, evaluation=Evaluation.anterior_wall
        )
        x = contour_subset[:, 0]
        y = contour_subset[:, 1]
        values = contour_subset[:, 2]

    # Filter out high curvature
    if filter_high_curvature:
        contour_subset = filter_out_high_curvature(x, y, values)
        x = contour_subset[:, 0]
        y = contour_subset[:, 1]
        values = contour_subset[:, 2]

    if normalize_by_motion:
        values = normalize_vs_by_motion(x, y, values, motion_map)

    ax.plot(values)
    return fig


# Get dataset and series
dataset = dev_dataset()
series_ids = get_series_ids(dataset)

# Load predictions and annotations
predictions_dict = load_all_predictions()
annotations = load_predictions(annotations_path)

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

sample = dataset[series_id]

# Select adhesion types
adhesion_types_select = st.sidebar.multiselect(
    "Adhesion types",
    ["anterior", "pelvis", "inside"],
)
adhesion_types = []
for adhesion_type in adhesion_types_select:
    if adhesion_type == "anterior":
        adhesion_types.append(AdhesionType.anteriorWall)
    if adhesion_type == "pelvis":
        adhesion_types.append(AdhesionType.pelvis)
    if adhesion_type == "inside":
        adhesion_types.append(AdhesionType.inside)

# Get metrics
frocs = []
slice_rocs = []
metrics_legend = []
for prediction_label in prediction_subset:
    metrics = picai_eval(
        predictions_dict[prediction_label],
        annotations,
        flat=True,
        types=adhesion_types,
    )
    frocs.append((metrics["FP_per_case"], metrics["sensitivity"]))
    slice_rocs.append((metrics["fpr"], metrics["tpr"], metrics["auroc"]))
    metrics_legend.append(prediction_label)

# Plot motion map
# motion_map = load_motion_map(sample)
# motion_fig = plt.figure()
# motion_fig, ax = plt.subplots()
# plot_frame(ax, sample["numpy"][0])
# plt.imshow(motion_map, alpha=1)

# Load specific prediction
prediction = load_prediction(predictions, series_id)
annotation = load_prediction(annotations, series_id)

# Toggle bounding boxes
label_boxes_toggle = st.sidebar.checkbox("Display ground truth boxes", value=True)
predicted_boxes_toggle = st.sidebar.checkbox("Display predicted boxes", value=True)

# Toggle filtering high curvature
# curvature_toggle = st.sidebar.checkbox("Filter high curvature", value=False)

# Toggle filtering prior
# prior_toggle = st.sidebar.checkbox("Filter contour prior", value=False)

# Toggle normalization by motion
# normalize_toggle = st.sidebar.checkbox("Normalize by motion", value=False)

# Layout
col1, col2 = st.columns(2)
with col1:
    movie = dataset[series_id]["numpy"]
    movie = movie / np.max(movie)
    fig = get_interactive_plotly_movie(movie)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.pyplot(
        plot_vs(
            series_id,
            prediction,
            annotation,
            adhesion_types,
            plot_label_boxes=label_boxes_toggle,
            plot_predicted_boxes=predicted_boxes_toggle,
            # filter_prior=prior_toggle,
            # filter_high_curvature=curvature_toggle,
            # normalize_by_motion=normalize_toggle,
            # motion_map=None,
        )
    )
    # st.pyplot(
    #     plot_vs_flat(
    #         series_id,
    #         filter_prior=prior_toggle,
    #         filter_high_curvature=curvature_toggle,
    #         normalize_by_motion=normalize_toggle,
    #         motion_map=motion_map,
    #     )
    # )
    st.pyplot(plot_ROCs(slice_rocs, legends=metrics_legend))
    st.pyplot(plot_FROCs(frocs, legends=metrics_legend))
