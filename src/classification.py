"""Classification from features, finally resulting in box predictions"""
from report_guided_annotation.extract_lesion_candidates import preprocess_softmax
import numpy as np
from src.adhesions import Adhesion


def get_boxes_from_raw(prediction, x, y, min_size=15):
    """Convert raw prediction to bounding boxes"""
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
