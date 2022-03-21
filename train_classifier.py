"""Train a Logistic Regression classifier with 5-fold CV, with features
from extract_features.py"""
from src.contour import Evaluation
from src.datasets import dev_dataset, folds
from src.classification import (
    load_pixel_features,
    get_feature_array,
    LogisticRegressionClassifier,
)
from pathlib import Path
from tqdm import tqdm

# Parameters
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
evaluation = Evaluation.pelvis

predictions_dict = {}
for fold_idx in tqdm(range(5), desc="Training classifiers"):
    train_series = folds[str(fold_idx)]["train"]
    train_features, train_labels = get_feature_array(
        features, train_series, evaluation=evaluation
    )

    # Fit classifier
    clf = LogisticRegressionClassifier()
    clf.fit(train_features, train_labels)
    clf.save(model_dir / f"model-{fold_idx}.joblib")
