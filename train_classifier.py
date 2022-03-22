"""Train a Logistic Regression classifier with 5-fold CV, with features
from extract_features.py"""
from src.contour import Evaluation
from src.datasets import dev_dataset, folds
from src.classification import (
    load_pixel_features,
    load_deep_features,
    get_feature_array,
    LogisticRegressionClassifier,
    MLP,
)
from pathlib import Path
from tqdm import tqdm

# Parameters
# included_features = ["slide", "percentage", "average_motion"]
included_features = ["slide", "average_motion", "max_motion", "local_motion", "deep"]
model_dir = Path("/home/bram/data/registration_method/models")

dataset = dev_dataset()
features = load_pixel_features()
deep_features = load_deep_features()
evaluation = {"anterior": Evaluation.anterior_wall, "pelvis": Evaluation.pelvis}

for region in evaluation:
    print(f"{region} region")
    for fold_idx in tqdm(range(5), desc="Training classifiers"):
        train_series = folds[str(fold_idx)]["train"]
        train_features, train_labels = get_feature_array(
            features,
            deep_features,
            train_series,
            evaluation=evaluation[region],
            included_features=included_features,
        )

        # Binarize labels
        if region == "anterior":
            train_labels = train_labels == 1
        if region == "pelvis":
            train_labels = train_labels == 2

        # Fit classifier
        # clf = LogisticRegressionClassifier()
        clf = MLP()
        clf.fit(train_features, train_labels)
        clf.save(model_dir / f"{region}-model-{fold_idx}.joblib")
