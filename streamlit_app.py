"""Visualize inference results from inference.py with streamlit"""
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import json
from cinemri.visualisation import plot_frame, get_interactive_plotly_movie
from cinemri.config import ARCHIVE_PATH, META_PATH
import numpy as np
import pickle
from src.contour import (
    filter_out_high_curvature,
    filter_out_prior_vs_subset,
    Evaluation,
)
from src.datasets import dev_dataset, frank_500_dataset
from src.adhesions import load_annotations, load_predictions, AdhesionType
from src.detection_pipeline import (
    plot_FROCs,
    plot_ROCs,
    compute_slice_level_ROC,
    froc_wrapper,
)
from src.vs_computation import calculate_motion_map
from src.evaluation import picai_eval
from src.classification import (
    load_pixel_features,
    load_deep_features,
    get_feature_array,
)

st.set_page_config(layout="wide")

inference_dir = Path("/home/bram/data/registration_method")
predictions_path = inference_dir / "predictions"
annotations_path = META_PATH / "bounding_boxes" / "first_frame_with_region.json"
raw_predictions_path = Path(
    "/home/bram/data/registration_method/predictions_raw/raw_predictions.json"
)
resnet_image_inference_dir = Path("/home/bram/data/registration_method/image_inference")
with open(raw_predictions_path) as json_file:
    raw_predictions = json.load(json_file)


def get_series_ids(dataset):
    numbered_series_ids = {}
    for idx, series_id in enumerate(dataset.series_instance_uids):
        numbered_series_ids[idx] = series_id
    return numbered_series_ids


def load_resnet_image_inference(series_id):
    for path in resnet_image_inference_dir.rglob(series_id + ".npy"):
        return np.load(path)


def load_feature(features, deep_features, series_id, feature_label, evaluation):
    """Load a 1D feature map for a series_id"""
    features, labels = get_feature_array(
        features,
        deep_features,
        [series_id],
        evaluation=evaluation,
        included_features=[feature_label],
    )
    return features[:, 0]


def load_raw_prediction(input_series_id, types):
    for patient_id in raw_predictions:
        for study_id in raw_predictions[patient_id]:
            for series_id, trial_pred_dict in raw_predictions[patient_id][
                study_id
            ].items():
                if series_id == input_series_id:
                    pred_dict = trial_pred_dict

    values = []
    x = []
    y = []
    for region in ["pelvis", "anterior"]:
        values += pred_dict[region]["prediction"]
        x += pred_dict[region]["x"]
        y += pred_dict[region]["y"]
    return values, x, y


def get_feature_labels(features):
    for series_id, feature_dict in features.items():
        return list(feature_dict.keys())


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


def plot_feature(
    sample,
    feature,
    x,
    y,
    plot_label_boxes,
    plot_predicted_boxes,
    prediction,
    title,
    limits=None,
):
    image = sample["numpy"][0]
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

    fig, ax = plt.subplots()
    plot_frame(ax, image, boxes=boxes)
    if limits is None:
        sc = ax.scatter(x, y, s=5, c=feature, cmap="jet")
    else:
        sc = ax.scatter(
            x, y, s=5, c=feature, cmap="jet", vmin=limits[0], vmax=limits[1]
        )
    plt.colorbar(sc)

    plt.title(title)
    return fig


def plot_image_prediction(
    sample,
    prediction,
    plot_label_boxes,
    title,
    limits=None,
):
    image = sample["numpy"][0]
    # Assemble ground truth boxes
    boxes = []
    if plot_label_boxes:
        for box in annotation:
            if box[2] not in adhesion_types:
                continue
            boxes.append({"box": box[0], "color": "green"})

    fig, ax = plt.subplots()
    plot_frame(ax, image, boxes=boxes)
    if limits is None:
        sc = ax.imshow(prediction, cmap="jet", alpha=0.3)
    else:
        sc = ax.imshow(
            prediction, cmap="jet", alpha=0.3, vmin=limits[0], vmax=limits[1]
        )
    plt.colorbar(sc)

    plt.title(title)
    return fig


# Get dataset and series
dataset = dev_dataset()
series_ids = get_series_ids(dataset)

# Load features
features = load_pixel_features()
deep_features = load_deep_features()
feature_labels = get_feature_labels(features)
feature_labels += ["combined"]
feature_labels += ["deep_output"]

# Load predictions and annotations
predictions_dict = load_all_predictions()
annotations = load_predictions(annotations_path)

# Select feature to visualize
feature_label = st.sidebar.selectbox("Feature visualization", feature_labels)

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

# Select adhesion types
adhesion_types_select = st.sidebar.multiselect(
    "Adhesion types", ["anterior", "pelvis", "inside"], default=["anterior"]
)
adhesion_types = []
for adhesion_type in adhesion_types_select:
    if adhesion_type == "anterior":
        adhesion_types.append(AdhesionType.anteriorWall)
    if adhesion_type == "pelvis":
        adhesion_types.append(AdhesionType.pelvis)
    if adhesion_type == "inside":
        adhesion_types.append(AdhesionType.inside)

# Set evaluation
if "anterior" in adhesion_types_select and "pelvis" in adhesion_types_select:
    evaluation = Evaluation.joint
else:
    if "anterior" in adhesion_types_select:
        evaluation = Evaluation.anterior_wall
    if "pelvis" in adhesion_types_select:
        evaluation = Evaluation.pelvis

# Load sample and features
sample = dataset[series_id]
if feature_label == "combined":
    slide = load_feature(features, deep_features, series_id, "slide", evaluation)
    motion = load_feature(
        features, deep_features, series_id, "local_motion", evaluation
    )
    feature_map = slide / motion
else:
    feature_map = load_feature(
        features, deep_features, series_id, feature_label, evaluation
    )
x = load_feature(features, deep_features, series_id, "x", evaluation)
y = load_feature(features, deep_features, series_id, "y", evaluation)
raw_prediction, x_raw, y_raw = load_raw_prediction(series_id, adhesion_types_select)
image_prediction = load_resnet_image_inference(series_id)

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
    pass


# Load specific prediction
prediction = load_prediction(predictions, series_id)
annotation = load_prediction(annotations, series_id)

# Toggle bounding boxes
label_boxes_toggle = st.sidebar.checkbox("Display ground truth boxes", value=True)
predicted_boxes_toggle = st.sidebar.checkbox("Display predicted boxes", value=True)

# Layout
col1, col2 = st.columns(2)
with col1:
    movie = dataset[series_id]["numpy"]
    movie = movie / np.max(movie)
    fig = get_interactive_plotly_movie(movie)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.pyplot(
        plot_feature(
            sample,
            raw_prediction,
            x_raw,
            y_raw,
            plot_label_boxes=label_boxes_toggle,
            plot_predicted_boxes=predicted_boxes_toggle,
            prediction=prediction,
            title="Raw prediction",
            limits=[0, 1],
        )
    )
    st.pyplot(
        plot_image_prediction(
            sample,
            image_prediction,
            plot_label_boxes=label_boxes_toggle,
            title="Full image prediction",
            limits=[0, 1],
        )
    )
    st.pyplot(
        plot_feature(
            sample,
            feature_map,
            x,
            y,
            plot_label_boxes=label_boxes_toggle,
            plot_predicted_boxes=False,
            prediction=prediction,
            title=feature_label,
        )
    )
    st.pyplot(plot_ROCs(slice_rocs, legends=metrics_legend))
    st.pyplot(plot_FROCs(frocs, legends=metrics_legend))
