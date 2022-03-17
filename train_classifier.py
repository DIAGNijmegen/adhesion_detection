"""Train a Logistic Regression classifier with 5-fold CV, with features
from extract_features.py"""
from src.contour import Evaluation
from src.datasets import dev_dataset, folds
from src.classification import (
    load_pixel_features,
    get_feature_array,
    get_normalizer,
)
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from tqdm import tqdm
from joblib import dump

# Parameters
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
evaluation = Evaluation.anterior_wall

predictions_dict = {}
for fold_idx in tqdm(range(5), desc="Training classifiers"):
    train_series = folds[str(fold_idx)]["train"]
    train_features, train_labels = get_feature_array(features, train_series)
    normalizer = get_normalizer(train_features)

    # Fit Logistic regression classifier
    clf = LogisticRegression().fit(train_features, train_labels)
    dump(clf, model_dir / f"model-{fold_idx}.joblib")
