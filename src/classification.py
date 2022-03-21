"""Classification from features, finally resulting in box predictions"""
from report_guided_annotation.extract_lesion_candidates import preprocess_softmax
from src.adhesions import Adhesion
from src.contour import Evaluation, get_adhesions_prior_coords
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from pathlib import Path
import pickle
from joblib import dump, load


def load_pixel_features():
    """Load pixel features for all series from disk"""
    feature_dataset_path = Path("/home/bram/data/registration_method/features.pkl")
    with open(feature_dataset_path, "r+b") as pkl_file:
        features = pickle.load(pkl_file)

    return features


def get_feature_array(
    features,
    series_ids,
    included_features=["slide", "x", "y"],
    evaluation=Evaluation.anterior_wall,
):
    """Constructs a feature array in sklearn format"""
    assembled = {}
    label = []
    mask = []
    for feature_label in included_features:
        assembled[feature_label] = []

    for series_id in series_ids:
        for feature_label in included_features:
            assembled[feature_label] += list(features[series_id][feature_label])
        label += list(features[series_id]["label"])

        # Get contour part masks
        x = features[series_id]["x"]
        y = features[series_id]["y"]
        case_mask = get_adhesions_prior_coords(
            x, y, evaluation=evaluation, return_mask=True
        )
        mask += list(case_mask)

    feature_array = np.zeros((len(label), len(included_features)))
    for idx, feature_label in enumerate(included_features):
        feature_array[:, idx] = assembled[feature_label]

    return feature_array[mask], np.array(label)[mask]


def get_boxes_from_raw(prediction, x, y, min_size=15):
    """Convert raw prediction to bounding boxes"""
    # Convert to np arrays
    prediction = np.array(prediction)
    x = np.array(x)
    y = np.array(y)

    all_hard_blobs, confidences, indexed_pred = preprocess_softmax(
        prediction, threshold="dynamic", min_voxels_detection=5
    )
    bounding_boxes = []
    for idx, confidence in confidences:
        x_lesion = x[indexed_pred == idx]
        y_lesion = y[indexed_pred == idx]
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
        self.classifier = LogisticRegression(class_weight="balanced")
