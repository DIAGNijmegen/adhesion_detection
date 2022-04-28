"""Classification from features, finally resulting in box predictions"""
from report_guided_annotation.extract_lesion_candidates import preprocess_softmax
from src.adhesions import Adhesion
from src.contour import Evaluation, get_adhesions_prior_coords
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pathlib import Path
import pickle
from joblib import dump, load
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt


def load_pixel_features():
    """Load pixel features for all series from disk"""
    feature_dataset_path = Path("/home/bram/data/registration_method/features.pkl")
    with open(feature_dataset_path, "r+b") as pkl_file:
        features = pickle.load(pkl_file)

    return features


def load_deep_features():
    """Load deep features for all series from disk"""
    feature_dataset_path = Path(
        "/home/bram/data/registration_method/resnet_features_49.pkl"
    )
    with open(feature_dataset_path, "r+b") as pkl_file:
        features = pickle.load(pkl_file)

    return features


def get_feature_array(
    features,
    deep_features,
    series_ids,
    included_features=["slide", "x", "y"],
    evaluation=Evaluation.anterior_wall,
):
    """Constructs a feature array in sklearn format"""
    assembled = {}
    label = []
    mask = []
    for feature_label in included_features:
        if feature_label == "deep":
            for idx in range(deep_features[series_ids[0]]["avgpool"].shape[1]):
                assembled[f"deep-{idx}"] = []
        else:
            assembled[feature_label] = []

    for series_id in series_ids:
        for feature_label in included_features:
            if feature_label == "deep":
                for idx in range(deep_features[series_id].shape[1]):
                    assembled[f"deep-{idx}"] += list(deep_features[series_id][:, idx])
            elif feature_label == "deep_output":
                assembled[feature_label] += list(deep_features[series_id]["output"])
            else:
                assembled[feature_label] += list(features[series_id][feature_label])
        label += list(features[series_id]["label"])

        # Get contour part masks
        x = features[series_id]["x"]
        y = features[series_id]["y"]
        case_mask = get_adhesions_prior_coords(
            x, y, evaluation=evaluation, return_mask=True
        )
        mask += list(case_mask)

    feature_array = np.zeros((len(label), len(assembled)))
    for idx, feature_label in enumerate(assembled):
        feature_array[:, idx] = assembled[feature_label]

    return feature_array[mask], np.array(label)[mask]


def order_by_contour(prediction, x_array, y_array):
    """Order the prediction and x, y arrays by contour. Sometimes it
    is possible that x and y are not in a logical order, going over the
    contour border in the middle of the arrays."""
    start_idx = 0
    for i in range(len(x_array) - 1):
        x, y = x_array[i], y_array[i]
        x_next, y_next = x_array[i + 1], y_array[i + 1]

        distance = np.sqrt((x - x_next) ** 2 + (y - y_next) ** 2)

        if distance > 1.5:
            start_idx = i + 1
            break

    if start_idx == 0:
        return prediction, x_array, y_array

    x_ordered = np.concatenate([x_array[start_idx:], x_array[:start_idx]])
    y_ordered = np.concatenate([y_array[start_idx:], y_array[:start_idx]])
    prediction_ordered = np.concatenate(
        [prediction[start_idx:], prediction[:start_idx]]
    )
    return prediction_ordered, x_ordered, y_ordered


def smooth_1d(prediction):
    window_size = 10
    if len(prediction) < window_size:
        window_size = len(prediction) - 5
    return savgol_filter(prediction, window_size, 3)


def get_boxes_from_raw(prediction, x, y, min_size=15):
    """Convert raw prediction to bounding boxes"""
    # Convert to np arrays
    prediction = np.array(prediction)
    x = np.array(x)
    y = np.array(y)

    # Smooth prediction
    prediction = smooth_1d(prediction)

    prediction_ordered, x_ordered, y_ordered = order_by_contour(prediction, x, y)

    all_hard_blobs, confidences, indexed_pred = preprocess_softmax(
        prediction_ordered,
        threshold="dynamic",
        min_voxels_detection=2,
        dynamic_threshold_factor=1.1,
        num_lesions_to_extract=5,
    )
    bounding_boxes = []
    for idx, confidence in confidences:
        x_lesion = x_ordered[indexed_pred == idx]
        y_lesion = y_ordered[indexed_pred == idx]
        x_box = np.min(x_lesion)
        y_box = np.min(y_lesion)
        w_box = np.max(x_lesion) - x_box
        h_box = np.max(y_lesion) - y_box

        # Adjust to minimum size
        if w_box < min_size:
            x_box = x_box - (min_size - w_box) // 2
            w_box = min_size
        if h_box < min_size:
            y_box = y_box - (min_size - h_box) // 2
            h_box = min_size

        adhesion = Adhesion([x_box, y_box, w_box, h_box])
        bounding_boxes.append((adhesion, confidence))

    return bounding_boxes


def get_segmentation_from_raw(prediction, x, y, image_size, min_size=15):
    """Convert raw prediction to line segmentations. These are detection
    maps as defined in picai_eval"""
    segmentation = np.zeros(image_size)

    # Convert to np arrays
    prediction = np.array(prediction)
    x = np.array(x)
    y = np.array(y)

    # Smooth prediction
    prediction = smooth_1d(prediction)

    prediction_ordered, x_ordered, y_ordered = order_by_contour(prediction, x, y)

    # Fill in prediction in segmentation
    for idx in range(len(prediction)):
        segmentation[int(y[idx]), int(x[idx])] = prediction[idx]

    segmentation = segmentation[..., None]

    # segmentation = binary_dilation(segmentation, structure=np.ones((3, 3)))

    all_hard_blobs, confidences, indexed_pred = preprocess_softmax(
        segmentation,
        threshold="dynamic",
        min_voxels_detection=2,
        dynamic_threshold_factor=1.1,
        num_lesions_to_extract=5,
    )

    return all_hard_blobs[:, :, 0]


def load_model(file_path):
    return load(file_path)


class BaseClassifier:
    def __init__(self):
        """Base class for a classifier that includes z-score
        normalization. self.classifier should be overwritten with a
        sklearn classifier.
        """
        self.normalizer = StandardScaler()
        self.classifier = None

    def fit(self, x, y):
        self.normalizer.fit(x)
        x_norm = self.normalizer.transform(x)
        self.classifier.fit(x_norm, y)

    def predict(self, x):
        x_norm = self.normalizer.transform(x)
        return self.classifier.predict_proba(x_norm)[:, 1]

    def save(self, file_path):
        dump(self, file_path)


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = LogisticRegression(
            class_weight="balanced", verbose=1, max_iter=1000
        )


class MLP(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = MLPClassifier(
            max_iter=2000, early_stopping=True, verbose=True
        )


class RandomForest(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = RandomForestClassifier(class_weight="balanced", verbose=True)
