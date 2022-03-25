"""A wrapper around the prediction/annotation format of this repository
and picai_eval's froc routine"""
import numpy as np
from prostatemr_evaluation.eval import evaluate
from .adhesions import AdhesionType
from copy import deepcopy


def box_list_to_image(box_list):
    """Given a box list in format List[(Adhesion, confidence)], convert
    to an image with boxes as rectangles. The value is confidence.

    Parameters
    ----------
    prediction_list: list of (Adhesion, confidence)
    """
    image = np.zeros((256, 192))
    for adhesion, confidence in box_list:
        x, y = int(adhesion.origin_x), int(adhesion.origin_y)
        w, h = int(adhesion.width), int(adhesion.height)
        image[y : y + h, x : x + w] = confidence
    return image[None, ...]


def filter_types(adhesion_dict, types):
    adhesion_dict_filtered = deepcopy(adhesion_dict)
    for series_id in adhesion_dict:
        adhesion_dict_filtered[series_id] = []
        for entry in adhesion_dict[series_id]:
            if entry[0].type not in types:
                continue
            adhesion_dict_filtered[series_id].append(entry)

    return adhesion_dict_filtered


def picai_eval(
    predictions,
    annotations,
    iou_threshold=0.001,
    flat=False,
    types=[AdhesionType.anteriorWall, AdhesionType.pelvis, AdhesionType.inside],
):
    """This converts bounding boxes to a mask with filled rectangles and
    then calls picai_eval's evaluate to get metrics"""
    annotations = filter_types(annotations, types)
    predictions = filter_types(predictions, types)
    subject_list = []
    predictions_list = []
    annotations_list = []
    for idx, series_id in enumerate(annotations):
        prediction = predictions[series_id]
        annotation = annotations[series_id]
        prediction_image = box_list_to_image(prediction)
        annotation_image = box_list_to_image(annotation)
        predictions_list.append(prediction_image)
        annotations_list.append(annotation_image)
        subject_list.append(series_id)

    metrics = evaluate(
        predictions_list,
        annotations_list,
        min_overlap=iou_threshold,
        subject_list=subject_list,
    )
    return metrics
