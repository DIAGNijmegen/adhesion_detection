"""A wrapper around the prediction/annotation format of this repository
and picai_eval's froc routine"""
import numpy as np
from prostatemr_evaluation.eval import (
    froc_from_lesion_evaluations,
    ap_from_lesion_evaluations,
)
from sklearn.metrics import roc_curve, auc
from .adhesions import AdhesionType
from copy import deepcopy


def evaluate_case(prediction_list, annotation_list, iou_threshold=0.1):
    """Gather the list of adhesion candidates, and classify in TP/FP/FN.

    Note: this adds 10 to all confidence scores, because picai_eval does
    not handle negative confidence scores

    Parameters
    ----------
    prediction_list: list of (Adhesion, confidence)
    annotation_list: list of (Adhesion, 1)
        Reference annotations/ground truth

    Returns
    -------
    y_list: list of (is_adhesion, confidence, iou)
    """
    y_list = []

    # If no annotations, all predictions are FP
    if len(annotation_list) == 0:
        # All predicted bounding boxes are FPs
        for adhesion, confidence in prediction_list:
            y_list.append((0, confidence + 10, 0))

        return y_list

    # Otherwise, for each adhesion in annotation_list, try to find match

    # Tracks which predictions has been assigned to a TP
    hit_predictions_inds = []
    # Loop over TPs
    for adhesion, _ in annotation_list:
        max_iou = 0
        max_iou_ind = -1
        # For simplicity for now one predicted bb can correspond to only one TP
        for ind, (bounding_box, confidence) in enumerate(prediction_list):
            curr_iou = adhesion.iou(bounding_box)
            # Do not use predictions that have already been assigned to a TP
            if curr_iou > max_iou and ind not in hit_predictions_inds:
                max_iou = curr_iou
                max_iou_ind = ind

        # If a maximum IoU is greater than the threshold, consider a TP as found
        if max_iou >= iou_threshold:
            y_list.append((1, confidence + 10, max_iou))
            hit_predictions_inds.append(max_iou_ind)
        # Otherwise add it as a false negative
        else:
            y_list.append((1, 0, 0))

    # Predictions that were not matched with a TP are FPs
    tp_mask = np.full(len(prediction_list), True, dtype=bool)
    tp_mask[hit_predictions_inds] = False
    fps_left = np.array(prediction_list)[tp_mask]
    for fp, confidence in fps_left:
        y_list.append((0, confidence + 10, 0))

    return y_list


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
    annotations = filter_types(annotations, types)
    predictions = filter_types(predictions, types)
    y_list = []
    roc_true = {}
    roc_pred = {}
    subject_list = []
    for idx, series_id in enumerate(annotations):
        prediction = predictions[series_id]
        annotation = annotations[series_id]
        y_list_case = evaluate_case(prediction, annotation, iou_threshold)

        # If no predictions and labels, pred is zero confidence
        if len(y_list_case) == 0:
            roc_true[series_id] = 0
            roc_pred[series_id] = 0
        else:
            roc_true[series_id] = np.max([a[0] for a in y_list_case])
            roc_pred[series_id] = np.max([a[1] for a in y_list_case])
        y_list += y_list_case
        subject_list.append(series_id)

    # Get adhesion-level results
    sensitivity, FP_per_case, thresholds, num_lesions = froc_from_lesion_evaluations(
        y_list=y_list, num_patients=len(annotations)
    )

    # Calculate recall, precision and average precision
    AP, precision, recall, _ = ap_from_lesion_evaluations(y_list, thresholds=thresholds)

    # Calculate case-level AUROC
    fpr, tpr, _ = roc_curve(
        y_true=[roc_true[s] for s in subject_list],
        y_score=[roc_pred[s] for s in subject_list],
        pos_label=1,
    )
    auc_score = auc(fpr, tpr)

    if flat:
        # flatten roc_true and roc_pred
        roc_true_flat = [roc_true[s] for s in subject_list]
        roc_pred_flat = [roc_pred[s] for s in subject_list]

    metrics = {
        "FP_per_case": FP_per_case,
        "sensitivity": sensitivity,
        "thresholds": thresholds,
        "num_lesions": num_lesions,
        "num_patients": len(annotations),
        "roc_true": (roc_true_flat if flat else roc_true),
        "roc_pred": (roc_pred_flat if flat else roc_pred),
        "AP": AP,
        "precision": precision,
        "recall": recall,
        # patient-level predictions
        "auroc": auc_score,
        "tpr": tpr,
        "fpr": fpr,
    }

    return metrics
