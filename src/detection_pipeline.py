import numpy as np
import SimpleITK as sitk
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from .adhesions import AdhesionType, Adhesion, load_annotations, vis_computed_cum_vs
from .config import *
from cinemri.definitions import CineMRISlice
from .utils import load_visceral_slides, binning_intervals, get_inspexp_frames
from .stat import bb_size_stat, get_vs_range
from .contour import (
    get_connected_regions,
    filter_out_prior_vs_subset,
    filter_out_high_curvature,
)
from .froc.deploy_FROC import y_to_FROC
from scipy import stats
from sklearn.metrics import roc_curve, auc
from .vs_definitions import VSExpectationNormType, Region, VSTransform
from enum import Enum, unique
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import matplotlib.cm as cm
from .vis_visceral_slide import plot_vs_distribution

PREDICTION_COLOR = (0, 0.8, 0.2)


@unique
class ConfidenceType(Enum):
    mean = 0
    min = 1


@unique
class EvaluationMetrics(Enum):
    froc_all = 0
    froc_negative = 1
    precision = 2
    recall = 3
    slice_roc = 4


def get_predicted_bb_ranges(bb_mean_size, bb_size_std, std_coef=1.96):
    bb_size_max = (
        bb_mean_size[0] + std_coef * bb_size_std[0],
        bb_mean_size[1] + std_coef * bb_size_std[1],
    )
    return bb_mean_size, bb_size_max


def bb_from_points(region, min_size):
    """
    Calculates bounding box based on the coordinates of the points in the region
    and mean annotations width and height

    Parameters
    ----------
    region : list of list
       A list of contour points represented as 2d arrays

    min_size : tuple of float
       Min size of a bounding box to be created. The first values is width and the second is height

    Returns
    -------
    adhesion : Adhesion
       A bounding box corresponding to the region
    """

    # axis=0: x, axis=1: y
    def compute_origin_and_len(region, min_size, axis=0):
        """Computes the origin and bb lenght by a give axis and set the lenght to the"""
        region_centre = region[round(len(region) / 2) - 1]

        axis_min = np.min([coord[axis] for coord in region])
        axis_max = np.max([coord[axis] for coord in region])
        length = axis_max - axis_min + 1

        if length < min_size[axis]:
            origin = region_centre[axis] - round(min_size[axis] / 2)
            length = min_size[axis]
        else:
            origin = axis_min

        return origin, length

    origin_x, width = compute_origin_and_len(region, min_size, axis=0)
    origin_y, height = compute_origin_and_len(region, min_size, axis=1)
    return Adhesion([origin_x, origin_y, width, height])


def bb_adjusted_region(region, bounding_box):
    """
    Finds a subset of points in the region that are inside the given bounding box
    and returns the region and its start and end indices
    """

    x_inside_bb = (bounding_box.origin_x <= region[:, 0]) & (
        region[:, 0] <= bounding_box.max_x
    )
    y_inside_bb = (bounding_box.origin_y <= region[:, 1]) & (
        region[:, 1] <= bounding_box.max_y
    )
    inside_bb = x_inside_bb & y_inside_bb
    adjusted_region = region[inside_bb]

    if len(adjusted_region) == 0:
        return [], None, None

    adjusted_region_start = adjusted_region[0, :2]
    adjusted_region_end = adjusted_region[-1, :2]

    # coords = region[:, :2]
    # start_index = np.where((coords == adjusted_region_start).all(axis=1))[0][0]
    # end_index = np.where((coords == adjusted_region_end).all(axis=1))[0][0]

    # Find start and end index
    found_start_idx = False
    found_end_idx = False
    for idx, val in enumerate(inside_bb):
        if val and not found_start_idx:
            start_index = idx
            found_start_idx = True

        if not val and found_start_idx:
            end_index = idx - 1
            found_end_idx = True
            break

    if not found_end_idx:
        end_index = idx

    if start_index > end_index:
        temp = start_index
        start_index = end_index
        end_index = temp

    return adjusted_region, start_index, end_index


def find_min_vs(regions):
    """
    Finds minimum VS value in all regions and returns a region containing it, its index and
    index of the min value inside the region
    """

    min_region_ind = None
    min_slide_value = np.inf
    # Find minimum across region
    for (region_ind, region) in enumerate(regions):
        region_min = np.min(region.values)
        if region_min < min_slide_value:
            min_region_ind = region_ind
            min_slide_value = region_min

    region_of_prediction = regions[min_region_ind]
    vs_value_min_ind = np.argmin(region_of_prediction.values)

    return region_of_prediction, min_region_ind, vs_value_min_ind


def adhesions_with_region_growing(
    regions,
    bb_size_min,
    bb_size_max,
    vs_range,
    conf_type=ConfidenceType.mean,
    region_growing_ind=None,
    min_region_len=5,
    lr=None,
    decrease_tolerance=np.inf,
):
    """
    Parameters
    ----------
    regions : list of Region
       A list of connected regions on the contour to predict adhesions at
    bb_size_min : tuple of float
       A minimum size of a predicted bounding box
    bb_size_max : tuple of float
       A maximum size of a predicted bounding box
    vs_max : float
       A maximum value of the visceral slide in the considered range
    region_growing_ind : float, default=2.5
       A coefficient to determine the maximum (minimum) visceral slide value for prediction
    min_region_len : int, default=5
       A minimum number of points in the region to be sufficient for prediction

    """

    bounding_boxes = []
    # While there are regions that are larger than  min_region_len
    while len(regions) > 0:
        region_of_prediction, min_region_ind, vs_value_min_ind = find_min_vs(regions)
        min_slide_value = region_of_prediction.values[vs_value_min_ind]
        if region_growing_ind is None:
            max_region_slide_value = (
                np.sqrt(min_slide_value)
                if min_slide_value < 1
                else min_slide_value ** 2
            )
        else:
            max_region_slide_value = (
                min_slide_value * region_growing_ind
                if min_slide_value > 0
                else min_slide_value / region_growing_ind
            )

        # Looking for the regions boundaries
        start_ind = end_ind = vs_value_min_ind
        start_ind_found = end_ind_found = False
        start_decrease_num = end_decrease_num = 0
        prediction_region = Region.from_point(region_of_prediction.points[start_ind])
        while not (start_ind_found and end_ind_found):
            # If the VS value at the previous index is too large or the regions size exceeds maximum,
            # do not change the start index
            # Otherwise add the point to the region
            if not start_ind_found:
                # Check if visceral slide at the previous index can be added to the region
                new_start_ind = max(0, start_ind - 1)
                start_value = region_of_prediction.values[new_start_ind]

                if start_value < region_of_prediction.values[start_ind]:
                    start_decrease_num += 1
                else:
                    start_decrease_num = 0

                if start_value < max_region_slide_value:
                    start_ind = new_start_ind
                    prediction_region.append_point(
                        region_of_prediction.points[start_ind]
                    )

                    start_ind_found = (
                        prediction_region.exceeded_size(bb_size_max)
                        or start_decrease_num == decrease_tolerance
                        or start_ind == 0
                    )
                else:
                    start_ind_found = True

            if not end_ind_found:
                # Same steps for the end of the regions
                new_end_ind = min(len(region_of_prediction.points) - 1, end_ind + 1)
                end_value = region_of_prediction.values[new_end_ind]

                if end_value < region_of_prediction.values[end_ind]:
                    end_decrease_num += 1
                else:
                    end_decrease_num = 0

                if end_value < max_region_slide_value:
                    end_ind = new_end_ind
                    prediction_region.append_point(region_of_prediction.points[end_ind])

                    end_ind_found = (
                        prediction_region.exceeded_size(bb_size_max)
                        or start_decrease_num == decrease_tolerance
                        or end_ind == (len(region_of_prediction.points) - 1)
                    )
                else:
                    end_ind_found = True

            # Stop if there are no points left in the regions
            if start_ind == 0 and end_ind == (len(region_of_prediction.points) - 1):
                start_ind_found = end_ind_found = True

        # Only predict the bounding box if the region is large enough
        if end_ind - start_ind >= min_region_len:
            # Generate bounding box from region
            bb_points = region_of_prediction.points[start_ind:end_ind]
            bounding_box = bb_from_points(bb_points, bb_size_min)

            # If the predicted bounding box is too small, enlarge it.
            # This way we want to prevent the method from outputting small bounding boxes
            adjusted_region, start_ind, end_ind = bb_adjusted_region(
                region_of_prediction.points, bounding_box
            )
            if lr is not None:
                negative_bb_features = extract_features(bounding_box, adjusted_region)
                confidence = lr.predict(np.array(negative_bb_features).reshape(1, -1))[
                    0
                ]
            else:
                # Take mean region vs as confidence
                confidence = (
                    np.mean(adjusted_region[:, 2])
                    if conf_type == ConfidenceType.mean
                    else min_slide_value
                )
                confidence = (vs_range[1] - confidence) / (vs_range[1] - vs_range[0])
            bounding_boxes.append((bounding_box, confidence))

        # Cut out bounding box region
        # Remove the region of prediction from the array
        del regions[min_region_ind]
        # Add regions around the cutoff if they are large enough
        region_before_bb = Region.from_points(region_of_prediction.points[:start_ind])
        if region_before_bb.exceeded_size(bb_size_min):
            regions.append(region_before_bb)

        region_after_bb = Region.from_points(
            region_of_prediction.points[(end_ind + 1) :]
        )
        if region_after_bb.exceeded_size(bb_size_min):
            regions.append(region_after_bb)

    return bounding_boxes


def predict_consecutive_minima(
    regions,
    bb_size_min,
    bb_size_max,
    vs_range,
    min_region_len=5,
    **kwargs,
):
    bounding_boxes = []

    # While there are regions that are larger than min_region_len
    while len(regions) > 0:
        # Find minimum visceral slide across regions
        region_of_prediction, min_region_ind, vs_value_min_ind = find_min_vs(regions)
        min_slide_value = region_of_prediction.values[vs_value_min_ind]

        # Predict box at the minimum of size bb_size_max
        # Confidence is the minimum vs value
        width, height = bb_size_max
        origin_x = region_of_prediction.x[vs_value_min_ind] - width // 2
        origin_y = region_of_prediction.y[vs_value_min_ind] - height // 2
        box = Adhesion([origin_x, origin_y, width, height])
        if len(region_of_prediction) >= min_region_len:
            bounding_boxes.append((box, 10 - min_slide_value))

        # Get indices that are inside predicted box, with double size
        # to avoid overlapping boxes
        origin_x = region_of_prediction.x[vs_value_min_ind] - width
        origin_y = region_of_prediction.y[vs_value_min_ind] - height
        box = Adhesion([origin_x, origin_y, width * 2, height * 2])
        adjusted_region, start_ind, end_ind = bb_adjusted_region(
            region_of_prediction.points, box
        )

        # Remove those indices from current regions

        # Remove the region of prediction from the array
        del regions[min_region_ind]

        # Add regions around the cutoff if they are large enough
        region_before_bb = Region.from_points(region_of_prediction.points[:start_ind])
        if region_before_bb.exceeded_size(bb_size_min):
            regions.append(region_before_bb)

        region_after_bb = Region.from_points(
            region_of_prediction.points[(end_ind + 1) :]
        )
        if region_after_bb.exceeded_size(bb_size_min):
            regions.append(region_after_bb)

    return bounding_boxes


def bb_with_threshold(
    vs,
    bb_size_min,
    bb_size_max,
    vs_range,
    pred_func=adhesions_with_region_growing,
    apply_contour_prior=False,
    apply_curvature_filter=False,
    conf_type=ConfidenceType.mean,
    region_growing_ind=None,
    min_region_len=5,
    lr=None,
):
    """
    Predicts adhesions with bounding boxes based on the values of the normalized visceral slide and the
    specified threshold level

    Parameters
    ----------
    vs : VisceralSlide
       A visceral slide to make prediction for
    bb_size_min : tuple of float
       The minimum size of a bounding box to be predicted
    bb_size_max : tuple of float
       The maximum size of a bounding box to be predicted
    vs_range : tuple of float
       A range of visceral slide values without outliers
    pred_func : func
       A function to make a prediction
    Returns
    -------
    bounding_boxes : list of Adhesion
       A list of bounding boxes predicted based on visceral slide values
    """

    x, y, slide_value = vs.x, vs.y, vs.values

    # Optionally remove parts of contour
    if apply_curvature_filter:
        contour_subset = filter_out_high_curvature(x, y, slide_value)
        x = contour_subset[:, 0]
        y = contour_subset[:, 1]
        slide_value = contour_subset[:, 2]
    if apply_contour_prior:
        contour_subset = filter_out_prior_vs_subset(x, y, slide_value)
        x = contour_subset[:, 0]
        y = contour_subset[:, 1]
        slide_value = contour_subset[:, 2]
    else:
        contour_subset = np.zeros((len(x), 3))
        contour_subset[:, 0] = x
        contour_subset[:, 1] = y
        contour_subset[:, 2] = slide_value

    # Remove the outliers
    contour_subset = np.array(
        [vs for vs in contour_subset if vs_range[0] <= vs[2] <= vs_range[1]]
    )
    # If no points are left, the VS values is too high in the regions where adhesions can be present
    if len(contour_subset) == 0:
        return []

    # Predict
    suitable_regions = get_connected_regions(contour_subset)
    # Convert to array of Region instances
    suitable_regions = [Region.from_points(region) for region in suitable_regions]
    bounding_boxes = pred_func(
        suitable_regions,
        bb_size_min,
        bb_size_max,
        vs_range,
        conf_type=conf_type,
        region_growing_ind=region_growing_ind,
        min_region_len=min_region_len,
        lr=lr,
    )
    return bounding_boxes


def predict(
    visceral_slides,
    annotations_dict,
    negative_vs_needed,
    conf_type=ConfidenceType.mean,
    region_growing_ind=None,
    min_region_len=5,
    lr=None,
):
    """
    Performs prediction by visceral slide threshold and evaluates it

    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       Visceral slides to predict adhesions
    annotations_dict : dict
       A dictionary of GT annotations in format "slice_full_id" : [adhesion : Adhesion]
    Returns
    -------
    predictions : dict
       A dictionary of predictions in format "slice_full_id" : [(adhesion : Adhesion, confidence: float)]
    """

    annotations = annotations_dict.values()

    # Average bounding box size
    bb_mean_size, bb_size_std = bb_size_stat(annotations)
    bb_size_min, bb_size_max = get_predicted_bb_ranges(bb_mean_size, bb_size_std)

    # Adjust annotations centers
    for vs in visceral_slides:
        if vs.full_id in annotations_dict:
            annotation = annotations_dict[vs.full_id]
            for adhesion in annotation.adhesions:
                if (
                    not adhesion.intersects_contour(vs.x, vs.y)
                    and annotation.full_id
                    != "CM0020_1.2.752.24.7.621449243.4474616_1.3.12.2.1107.5.2.30.26380.2019060311131653190024186.0.0.0"
                ):
                    new_center = adhesion.contour_point_closes_to_center(
                        np.column_stack((vs.x, vs.y))
                    )
                    adhesion.adjust_center(new_center)

    # vary threshold level
    # Get predictions by visceral slide level threshold
    vs_range = get_vs_range(visceral_slides, negative_vs_needed)

    predictions = {}
    for vs in visceral_slides:
        # vs.zeros_fix()
        prediction = bb_with_threshold(
            vs,
            bb_size_min,
            bb_size_max,
            vs_range,
            adhesions_with_region_growing,
            conf_type,
            region_growing_ind,
            min_region_len,
            lr,
        )
        predictions[vs.full_id] = prediction

    return predictions


def evaluate(predictions, annotations_dict, output_path, iou_threshold=0.01):
    """
    Evaluates the predicted adhesions
    Parameters
    ----------
    predictions : dict
       A dictionary of predictions in format "slice_full_id" : [(adhesion : Adhesion, confidence: float)]
    annotations_dict : dict
       A dictionary of GT annotations in format "slice_full_id" : [adhesion : Adhesion]
    output_path : Path
       A path where to save visualised evaluation metrics
    iou_threshold : float, default=0.01
       IoU threshold to determine the hit
    """
    output_path.mkdir(exist_ok=True, parents=True)

    tps, fps, fns = get_prediction_outcome(annotations_dict, predictions, iou_threshold)
    negative_slices_ids = [
        full_id for full_id in predictions.keys() if full_id not in annotations_dict
    ]
    outcomes, outcomes_negative = get_confidence_outcome(
        tps, fps, fns, negative_slices_ids
    )

    # FROC
    slices_num = len(predictions)
    negative_slices_num = len(negative_slices_ids)
    fp_per_image, fp_per_negative_image, sensitivity, thresholds = y_to_FROC(
        outcomes, outcomes_negative, slices_num, negative_slices_num
    )
    froc = fp_per_image, fp_per_negative_image, sensitivity

    # Precision/Recall
    precision, recall, thresholds = compute_pr_curves(outcomes)
    ap, precision1, recall1 = compute_ap(recall, precision)
    pr_curves = precision, recall, thresholds
    print("Average precision {}".format(ap))

    # Slice level data
    slice_roc = compute_slice_level_ROC(predictions, annotations_dict)
    print("Slice level AUC {}".format(slice_roc[2]))

    # Confidence statistics
    tp_conf = [tp[2] for tp in tps]
    mean_tp_conf = np.mean(tp_conf)
    print("Mean TP conf {}".format(mean_tp_conf))

    fp_conf = [fp[2] for fp in fps]
    mean_fp_conf = np.mean(fp_conf)
    print("Mean FP conf {}".format(mean_fp_conf))

    t_test = stats.ttest_ind(tp_conf, fp_conf, equal_var=False)
    print("T-stat {}, p-value {}".format(t_test.statistic, t_test.pvalue))

    conf_stat = mean_tp_conf, mean_fp_conf, t_test.pvalue

    # Save metrics
    save_evaluation_metrics(froc, pr_curves, ap, slice_roc, conf_stat, output_path)

    # Plot metrics
    plot_evaluation_metrics(froc, pr_curves, slice_roc, output_path)


def save_evaluation_metrics(froc, pr_curves, ap, slice_roc, conf_stat, output_path):
    froc_dict = {"fp_all": froc[0], "fp_neg": froc[1], "sens": froc[2]}
    pr_curves_dict = {
        "precision": pr_curves[0],
        "recall": pr_curves[1],
        "thresholds": pr_curves[2],
    }
    slice_roc_dict = {"fpr": slice_roc[0], "tpr": slice_roc[1], "auc": slice_roc[2]}
    conf_dict = {
        "mean_tp": conf_stat[0],
        "mean_fp": conf_stat[1],
        "p_value": conf_stat[2],
    }

    metrics_dict = {
        "froc": froc_dict,
        "pr_curves": pr_curves_dict,
        "ap": ap,
        "slice_roc": slice_roc_dict,
        "conf_stat": conf_dict,
    }

    output_file = output_path / EVALUATION_METRICS_FILE
    with open(output_file, "wb") as f:
        pickle.dump(metrics_dict, f)


def load_evaluation_metrics(file_path):

    with open(file_path, "rb") as f:
        metrics_dict = pickle.load(f)

        froc_dict = metrics_dict["froc"]
        froc = froc_dict["fp_all"], froc_dict["fp_neg"], froc_dict["sens"]

        pr_curves_dict = metrics_dict["pr_curves"]
        pr_curves = (
            pr_curves_dict["precision"],
            pr_curves_dict["recall"],
            pr_curves_dict["thresholds"],
        )

        slice_roc_dict = metrics_dict["slice_roc"]
        slice_roc = slice_roc_dict["fpr"], slice_roc_dict["tpr"], slice_roc_dict["auc"]

        conf_dict = metrics_dict["conf_stat"]
        conf_stat = conf_dict["mean_tp"], conf_dict["mean_fp"], conf_dict["p_value"]

        ap = metrics_dict["ap"]

    return froc, pr_curves, slice_roc, ap, conf_stat


def plot_evaluation_metrics(froc, pr_curves, slice_roc, output_path):

    plot_FROCs([(froc[0], froc[2])], output_path, "FROC all")
    plot_FROCs([(froc[1], froc[2])], output_path, "FROC negative")

    plot_precisions_recalls([(pr_curves[0], pr_curves[2])], output_path)
    plot_precisions_recalls([(pr_curves[1], pr_curves[2])], output_path, "Recall")

    plot_ROCs([slice_roc], output_path=output_path)


def get_prediction_outcome(annotations_dict, predictions, iou_threshold=0.1):
    """
    Determines TPs, FPS and FNs based on the GT annotations, predictions and IoU threshold
    Parameters
    ----------
    annotations_dict : dict
       A dictionary of GT annotations in format "slice_full_id" : [adhesion : Adhesion]
    predictions : list of list
       Bounding boxes and confidences
    iou_threshold : float, default=0.01
       IoU threshold to determine the hit

    Returns
    -------
    tps, fps : list of tuple
       A list of predicted TP(FP)s in format (slice_id, Adhesion, confidence)
    fns : list of tuple
       A list of predicted FNs in format (slice_id, confidence)
    """
    tps = []
    fps = []
    fns = []

    for slice_id, prediction in predictions.items():
        if (
            slice_id
            == "CM0173_1.2.752.24.7.621449243.4290058_1.3.12.2.1107.5.2.30.26380.2018112713410524086741904.0.0.0"
        ):
            print("WATCH IT")
        else:
            print("ignore")
        if slice_id in annotations_dict:
            annotation = annotations_dict[slice_id]
            # Tracks which predictions has been assigned to a TP
            hit_predictions_inds = []
            # Loop over TPs
            for adhesion in annotation.adhesions:
                print(adhesion)
                max_iou = 0
                max_iou_ind = -1
                # For simplicity for now one predicted bb can correspond to only one TP
                for ind, (bounding_box, _) in enumerate(prediction):
                    curr_iou = adhesion.iou(bounding_box)
                    # Do not use predictions that have already been assigned to a TP
                    if curr_iou > max_iou and ind not in hit_predictions_inds:
                        max_iou = curr_iou
                        max_iou_ind = ind

                # If a maximum IoU is greater than the threshold, consider a TP as found
                if max_iou >= iou_threshold:
                    matching_pred = prediction[max_iou_ind]
                    tps.append((slice_id,) + matching_pred)
                    hit_predictions_inds.append(max_iou_ind)
                # Otherwise add it as a false negative
                else:
                    fns.append((slice_id, adhesion))

            # Get false positive
            # Predictions that was not matched with a TP are FPs
            tp_mask = np.full(len(prediction), True, dtype=bool)
            tp_mask[hit_predictions_inds] = False
            fps_left = np.array(prediction)[tp_mask]
            fps += [(slice_id, fp[0], fp[1]) for fp in fps_left]
        else:
            # All predicted bounding boxes are FPs
            fps += [(slice_id, fp[0], fp[1]) for fp in prediction]

    return tps, fps, fns


def get_confidence_outcome(tps, fps, fns, negative_ids):
    """
    Determines whether prediction with a given confidence is true or false based on the TPs and FPs lists
    Parameters
    ----------
    tps, fps : list of tuple
       A list of predicted TP(FP)s in format (slice_id, Adhesion, confidence)
    fns : list of tuple
       A list of predicted FNs in format (slice_id, confidence)
    negative_ids : list of str
       A list of negative ids

    Returns
    -------
    outcomes : list
       A list of tuple of confidence and whether its prediction is true
    outcomes_negative : list
       A list of tuple of confidence and whether its prediction is true for negative slices only
    """
    outcomes = []
    outcomes_negative = []
    for _, _, confidence in tps:
        outcomes.append((1, confidence))

    for slice_id, _, confidence in fps:
        outcomes.append((0, confidence))

        if slice_id in negative_ids:
            outcomes_negative.append((0, confidence))

    for _ in fns:
        outcomes.append((1, 0))

    return outcomes, outcomes_negative


def compute_slice_level_ROC(predictions, annotations_dict):

    thresholds = []
    labels = []
    for slice_full_id, prediction in predictions.items():
        if len(prediction) > 0:
            confidences = []
            for box, confidence in prediction:
                confidences.append(confidence)
            thresholds.append(np.max(confidences))
        else:
            thresholds.append(0)

        label = 1 if slice_full_id in annotations_dict else 0
        labels.append(label)

    fpr, tpr, _ = roc_curve(labels, thresholds)
    auc_val = auc(fpr, tpr)

    return fpr, tpr, auc_val


def compute_pr_curves(outcomes):
    """
    Computes precision and recall curves from list of predictions and confidence
    """

    outcomes.sort()
    labels = []
    predictions = []

    recall = []
    precision = []

    for label, prediction in outcomes:
        labels.append(label)
        predictions.append(prediction)

    # Total Number of Lesions
    y_true_all = np.array(labels)
    y_pred_all = np.array(predictions)
    total_lesions = y_true_all.sum()

    thresholds = np.unique(y_pred_all)
    thresholds.sort()
    thresholds = thresholds[::-1]

    for th in thresholds:
        if th > 0:
            y_pred_all_thresholded = np.zeros_like(y_pred_all)
            y_pred_all_thresholded[y_pred_all > th] = 1
            tp = np.sum(y_true_all * y_pred_all_thresholded)
            fp = np.sum(y_pred_all_thresholded - y_true_all * y_pred_all_thresholded)

            # Add the corresponding precision and recall values
            recall.append(tp / total_lesions)
            curr_precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
            precision.append(curr_precision)
        else:
            # Extend precision and recall curves
            if len(recall) > 0:
                recall.append(recall[-1])
                precision.append(precision[-1])

    return precision, recall, thresholds


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    Taken from https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap, mpre, mrec


def plot_ROCs(rocs_data, output_path=None, legends=None, colors=None):
    """
    Plots a slice-level ROC
    """
    plt.figure()
    if legends is not None:
        if colors is not None:
            for roc_data, legend, color in zip(rocs_data, legends, colors):
                plt.plot(
                    roc_data[0],
                    roc_data[1],
                    lw=2,
                    label="{}, AUC = {:.2f}".format(legend, roc_data[2]),
                    color=color,
                )
        else:
            for roc_data, legend in zip(rocs_data, legends):
                plt.plot(
                    roc_data[0],
                    roc_data[1],
                    lw=2,
                    label="{}, AUC = {:.2f}".format(legend, roc_data[2]),
                )
    else:
        for roc_data in rocs_data:
            plt.plot(
                roc_data[0], roc_data[1], lw=2, label="AUC = {:.2f}".format(roc_data[2])
            )

    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("Sensitivity")
    if legends is not None:
        plt.legend(loc="lower right")
    if output_path is not None:
        plt.savefig(output_path / "Slice_ROC.png", bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_FROCs(frocs, output_path, title, legends=None, colors=None):

    plt.figure()
    if legends is not None:
        if colors is not None:
            for froc, legend, color in zip(frocs, legends, colors):
                plt.plot(froc[0], froc[1], lw=2, label=legend, color=color)
        else:
            for froc, legend in zip(frocs, legends):
                plt.plot(froc[0], froc[1], lw=2, label=legend)
    else:
        for froc in frocs:
            plt.plot(froc[0], froc[1])

    plt.xlabel("Mean number of FPs per image")
    plt.ylabel("Sensitivity")
    plt.ylim([0, 1])
    plt.xscale("log")
    if legends is not None:
        plt.legend(loc="upper left")
    plt.savefig(output_path / "{}.png".format(title), bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_precisions_recalls(
    curves, output_path, metric="Precision", legends=None, colors=None
):

    plt.figure()
    if legends is not None:
        if colors is not None:
            for curve, legend, color in zip(curves, legends, colors):
                plt.plot(curve[1], curve[0], label=legend, color=color)
        else:
            for curve, legend in zip(curves, legends):
                plt.plot(curve[1], curve[0], label=legend)
    else:
        for curve in curves:
            plt.plot(curve[1], curve[0])

    plt.xlabel("Confidence")
    plt.ylabel(metric)
    if legends is not None:
        loc = "upper left" if metric == "Precision" else "upper right"
        plt.legend(loc=loc)
    if metric == "Recall":
        plt.ylim([0, 1])
    plt.savefig(
        output_path / "{}.png".format(metric), bbox_inches="tight", pad_inches=0
    )
    plt.show()


def visualize_gt_vs_prediction(
    prediction, annotation, x, y, values, insp_frame, file_path=None
):
    """
    Visualises ground truth annotations vs prediction together with visceral slide
    on the inspiration frame

    Parameters
    ----------
    annotation : AdhesionAnnotation
        A ground truth annotation for a slice
    prediction : list of (Adhesion, float)
       A list of predicted bounding boxes and the corresponding confidence scores
    x, y : ndarray of int
       x-axis and y-axis components of abdominal cavity contour
    values : ndarray of float
       Values of visceral slide corresponding to the coordinates of abdominal cavity contour
    insp_frame : ndarray
       An inspiration frame
    file_path : Path, optional
       A Path where to save a file, default is None
    """

    plt.figure()
    plt.imshow(insp_frame, cmap="gray")
    # Plot visceral slide
    plt.scatter(x, y, s=5, c=values, cmap="jet")
    plt.colorbar()
    ax = plt.gca()

    for index, (adhesion, confidence) in enumerate(prediction):
        # if index > 1:
        #    break
        adhesion_rect = Rectangle(
            (adhesion.origin_x, adhesion.origin_y),
            adhesion.width,
            adhesion.height,
            linewidth=1.5,
            edgecolor=PREDICTION_COLOR,
            facecolor="none",
        )
        ax.add_patch(adhesion_rect)
        plt.text(
            adhesion.origin_x,
            adhesion.origin_y - 3,
            "{:.3f}".format(confidence),
            c=PREDICTION_COLOR,
            fontweight="semibold",
        )

    """
        if annotation:
        for adhesion in annotation.adhesions:
            adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(adhesion_rect)
    """

    plt.axis("off")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()

    plt.close()


# TODO: probably leave only prior coords when plotting
def visualize(
    visceral_slides,
    annotations_dict,
    predictions,
    images_path,
    output_path,
    prior=False,
    inspexp_data=None,
):
    """

    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       Visceral slides to predict adhesions
    annotations_dict : dict
       A dictionary of GT annotations in format "slice_full_id" : [adhesion : Adhesion]
    predictions : dict
       A dictionary of predictions in format "slice_full_id" : [(adhesion : Adhesion, confidence: float)]
    images_path : Path
       A path to CineMRI slices
    output_path : Path
       A path where to save visualised prediction vs GT
    inspexp_data : dict, optional
       A dictionary with inspiration/ expiration frames ids data
    """

    output_path.mkdir(exist_ok=True, parents=True)

    for vs in visceral_slides:
        prediction = predictions[vs.full_id]

        if inspexp_data is None:
            # Assume it is cumulative VS and take the one before the last one frame
            slice_path = vs.build_path(images_path)
            slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
            frame = slice[-2]
        else:
            # Extract inspiration frame
            try:
                slice = CineMRISlice(vs.slice_id, vs.patient_id, vs.study_id)
                frame, _ = get_inspexp_frames(slice, inspexp_data, images_path)
            except:
                print(
                    "Missing insp/exp data for the patient {}, study {}, slice {}".format(
                        vs.patient_id, vs.study_id, vs.slice_id
                    )
                )

        file_path = output_path / (vs.full_id + ".png")
        annotation = (
            annotations_dict[vs.full_id] if vs.full_id in annotations_dict else None
        )

        if prior:
            x, y, slide_value = vs.x, vs.y, vs.values
            prior_subset = filter_out_prior_vs_subset(x, y, slide_value)
            x, y, values = prior_subset[:, 0], prior_subset[:, 1], prior_subset[:, 2]
        else:
            x, y, values = vs.x, vs.y, vs.values
        visualize_gt_vs_prediction(
            prediction, annotation, x, y, values, frame, file_path
        )


# TODO: document
# Plots normalized visceral slide values against adhesion frequency
def vs_adhesion_likelihood(
    visceral_slide_path,
    annotations_path,
    intervals_num=1000,
    adhesion_types=[AdhesionType.anteriorWall, AdhesionType.pelvis],
    plot=False,
):

    annotations_dict = load_annotations(
        annotations_path, as_dict=True, adhesion_types=adhesion_types
    )
    visceral_slides = load_visceral_slides(visceral_slide_path)
    vs_max = 0
    for visceral_slide in visceral_slides:
        vs_max = max(vs_max, np.max(visceral_slide.values))

    # Binning
    reference_vals = binning_intervals(end=vs_max, n=intervals_num)

    # Initialize frequencies
    freq_adh_dict = {}
    freq_not_adh_dict = {}
    for i in range(intervals_num):
        freq_adh_dict[i] = 0
        freq_not_adh_dict[i] = 0

    points_in_annotations = 0
    points_not_in_annotations = 0

    # Calculate frequencies in
    for vs in visceral_slides:
        has_annotation = vs.full_id in annotations_dict

        if not has_annotation:
            for value in vs.values:
                diff = reference_vals - value
                index = np.argmin(np.abs(diff))
                freq_not_adh_dict[index] = freq_not_adh_dict[index] + 1
                points_not_in_annotations += 1
        else:
            annotation = annotations_dict[vs.full_id]
            # For each point check if it intersects any adhesion
            for x, y, vs in zip(vs.x, vs.y, vs.values):
                intersects = False
                for adhesion in annotation.adhesions:
                    intersects = adhesion.contains_point(x, y)
                    if intersects:
                        break

                # Find the reference value closest to VS and increase the counter
                diff = reference_vals - vs
                index = np.argmin(np.abs(diff))

                if intersects:
                    freq_adh_dict[index] = freq_adh_dict[index] + 1
                    points_in_annotations += 1
                else:
                    freq_not_adh_dict[index] = freq_not_adh_dict[index] + 1
                    points_not_in_annotations += 1

    # Now extract ordered frequencies
    adh_frequencies = []
    not_adh_frequencies = []
    for i in range(intervals_num):
        adh_frequencies.append(freq_adh_dict[i])
        not_adh_frequencies.append(freq_not_adh_dict[i])

    # Divide frequencies to number of points to represent a likelihood
    adh_frequencies = np.array(adh_frequencies)
    adh_likelihood = adh_frequencies / points_in_annotations

    not_adh_frequencies = np.array(not_adh_frequencies)
    not_adh_likelihood = not_adh_frequencies / points_not_in_annotations

    if plot:
        max_y = max(adh_likelihood.max(), not_adh_likelihood.max()) * 1.1

        plt.figure()
        plt.plot(reference_vals, adh_likelihood)
        plt.axhline(y=0.001, color="r", linestyle="--")
        plt.ylim([0, max_y])
        plt.title("Adhesion")
        plt.xlabel("Normalised visceral slide")
        plt.ylabel("Likelihood")
        plt.savefig(
            "adh_freq_int_{}".format(intervals_num), bbox_inches="tight", pad_inches=0
        )
        plt.show()

        plt.figure()
        plt.plot(reference_vals, not_adh_likelihood)
        plt.axhline(y=0.001, color="r", linestyle="--")
        plt.ylim([0, max_y])
        plt.title("Not Adhesion")
        plt.xlabel("Normalised visceral slide")
        plt.ylabel("Likelihood")
        plt.savefig(
            "not_adh_freq_int_{}".format(intervals_num),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()

    return reference_vals, adh_likelihood, not_adh_likelihood


def vs_value_distr(visceral_slide_path, intervals_num=1000):

    visceral_slides = load_visceral_slides(visceral_slide_path)
    vs_max = 0
    for visceral_slide in visceral_slides:
        vs_max = max(vs_max, np.max(visceral_slide.values))

    # Binning
    reference_vals = binning_intervals(end=vs_max, n=intervals_num)

    # Initialize frequencies
    freq_dict = {}
    for i in range(intervals_num):
        freq_dict[i] = 0

    # Calculate frequencies in
    for visceral_slide in visceral_slides:
        for value in visceral_slide.values:
            diff = reference_vals - value
            index = np.argmin(np.abs(diff))
            freq_dict[index] = freq_dict[index] + 1

    frequencies = []
    for i in range(intervals_num):
        frequencies.append(freq_dict[i])

    plt.figure()
    plt.plot(reference_vals, frequencies)
    plt.title("VS values frequencies")
    plt.xlabel("Normalised visceral slide")
    plt.ylabel("Frequency")
    plt.savefig(
        "vs_freq_int_{}".format(intervals_num), bbox_inches="tight", pad_inches=0
    )
    plt.show()


def vs_adhesion_boxplot(
    visceral_slides,
    annotations_path,
    output_path,
    adhesion_types=[AdhesionType.anteriorWall, AdhesionType.pelvis],
    prior_only=False,
):

    output_path.mkdir(exist_ok=True, parents=True)

    annotations_dict = load_annotations(
        annotations_path, as_dict=True, adhesion_types=adhesion_types
    )

    points_in_annotations = []
    points_not_in_annotations = []

    # Calculate frequencies in
    for vs in visceral_slides:
        has_annotation = vs.full_id in annotations_dict
        if prior_only:
            x, y, slide_value = vs.x, vs.y, vs.values
            contour_subset = filter_out_prior_vs_subset(x, y, slide_value)
        else:
            contour_subset = np.column_stack((vs.x, vs.y, vs.values))

        if not has_annotation:
            points_not_in_annotations.extend(contour_subset[:, 2])
        else:
            annotation = annotations_dict[vs.full_id]

            # Adjust annotations centers if necessary
            for adhesion in annotation.adhesions:
                if (
                    not adhesion.intersects_contour(
                        contour_subset[:, 0], contour_subset[:, 1]
                    )
                    and annotation.full_id
                    != "CM0020_1.2.752.24.7.621449243.4474616_1.3.12.2.1107.5.2.30.26380.2019060311131653190024186.0.0.0"
                ):
                    new_center = adhesion.contour_point_closes_to_center(
                        contour_subset[:, :2]
                    )
                    adhesion.adjust_center(new_center)

                adhesion.adjust_size()

            # For each point check if it intersects any adhesion
            for coord in contour_subset:
                intersects = False
                for adhesion in annotation.adhesions:
                    intersects = adhesion.contains_point(coord[0], coord[1])
                    if intersects:
                        break

                if intersects:
                    points_in_annotations.append(coord[2])
                else:
                    points_not_in_annotations.append(coord[2])

    # Boxplot
    title = "adh_boxplot_prior" if prior_only else "adh_boxplot_all"
    plt.figure()
    plt.boxplot(points_in_annotations)
    plt.savefig(output_path / title, bbox_inches="tight", pad_inches=0)
    plt.show()

    # Histogram
    title = "adh_hist_prior" if prior_only else "adh_hist_all"
    plt.figure()
    plt.hist(points_in_annotations, bins=50)
    plt.savefig(output_path / title, bbox_inches="tight", pad_inches=0)
    plt.show()

    print("Adhesions VS stat:")
    print("Median {}".format(np.median(points_in_annotations)))
    print("25% precentile {}".format(np.percentile(points_in_annotations, 25)))
    print("75% precentile {}".format(np.percentile(points_in_annotations, 75)))

    # Boxplot
    title = "no_adh_boxplot_prior" if prior_only else "no_adh_boxplot_all"
    plt.figure()
    plt.boxplot(points_not_in_annotations)
    plt.savefig(output_path / title, bbox_inches="tight", pad_inches=0)
    plt.show()

    # Histogram
    title = "no_adh_hist_prior" if prior_only else "no_adh_hist_all"
    plt.figure()
    plt.hist(points_not_in_annotations, bins=200)
    plt.savefig(output_path / title, bbox_inches="tight", pad_inches=0)
    plt.show()

    print("No Adhesions VS stat:")
    print("Median {}".format(np.median(points_not_in_annotations)))
    print("25% precentile {}".format(np.percentile(points_not_in_annotations, 25)))
    print("75% precentile {}".format(np.percentile(points_not_in_annotations, 75)))

    t_test = stats.ttest_ind(
        points_in_annotations, points_not_in_annotations, equal_var=False
    )
    print("T-stat {}, p-value {}".format(t_test.statistic, t_test.pvalue))


def extract_features(adhesion_bb, adhesion_region):
    vs_values_region = adhesion_region[:, 2]  # np.mean(vs_values_region)
    adhesion_features = [1, adhesion_bb.height, adhesion_bb.width, len(adhesion_region)]
    return adhesion_features


def trainLR(visceral_slides, annotations_dict):

    # Get bounding boxes stat
    annotations = annotations_dict.values()
    bb_mean_size, bb_size_std = bb_size_stat(annotations)

    # Adjust annotations centers to account for different abdominal cavity boundary position
    for vs in visceral_slides:
        if vs.full_id in annotations_dict:
            annotation = annotations_dict[vs.full_id]
            for adhesion in annotation.adhesions:
                if (
                    not adhesion.intersects_contour(vs.x, vs.y)
                    and annotation.full_id
                    != "CM0020_1.2.752.24.7.621449243.4474616_1.3.12.2.1107.5.2.30.26380.2019060311131653190024186.0.0.0"
                ):
                    new_center = adhesion.contour_point_closes_to_center(
                        np.column_stack((vs.x, vs.y))
                    )
                    adhesion.adjust_center(new_center)

    positive_samples = []
    negative_samples = []
    for ind, vs in enumerate(visceral_slides):
        if vs.full_id in annotations_dict:
            annotation = annotations_dict[vs.full_id]
            # get VS region
            vs_region = np.column_stack((vs.x, vs.y, vs.values))
            for adhesion in annotation.adhesions:
                if adhesion.intersects_contour(vs.x, vs.y):
                    adhesion_region, _, _ = bb_adjusted_region(vs_region, adhesion)
                    adhesion_features = extract_features(adhesion, adhesion_region)
                    positive_samples.append(adhesion_features)
        else:
            # Sample negative bbs
            # get VS prior region
            x, y, slide_value = vs.x, vs.y, vs.values
            vs_region = filter_out_prior_vs_subset(x, y, slide_value)
            samples_num = 0
            if vs_region.shape[0] > 60:
                while samples_num < 4:
                    # Get center
                    center_ind = np.random.choice(
                        range(30, vs_region.shape[0] - 30), 1
                    )[0]
                    bb_center = vs_region[center_ind]
                    # get bb width and height
                    bb_width = int(
                        np.round(
                            bb_mean_size[0]
                            + bb_size_std[0] * np.random.normal(size=1)[0]
                        )
                    )
                    bb_height = int(
                        np.round(
                            bb_mean_size[1]
                            + bb_size_std[1] * np.random.normal(size=1)[0]
                        )
                    )
                    origin_x = int(bb_center[0] - round(bb_width / 2))
                    origin_y = int(bb_center[1] - round(bb_height / 2))
                    negative_bb = Adhesion([origin_x, origin_y, bb_width, bb_height])

                    adhesion_region, _, _ = bb_adjusted_region(vs_region, negative_bb)
                    if len(adhesion_region) > 0:
                        samples_num += 1
                        negative_bb_features = extract_features(
                            negative_bb, adhesion_region
                        )
                        negative_samples.append(negative_bb_features)

    samples = np.concatenate((negative_samples, positive_samples))
    labels = np.concatenate(
        (np.zeros(len(negative_samples)), np.ones(len(positive_samples)))
    )

    clf = sm.Logit(labels, samples).fit()
    print(clf.summary2())
    thresholds = clf.predict(samples)
    prediction = list(map(round, thresholds))
    print("Accuracy = ", accuracy_score(labels, prediction))

    fpr, tpr, _ = roc_curve(labels, thresholds)
    auc_val = auc(fpr, tpr)
    plot_ROCs([(fpr, tpr, auc_val)])

    return clf


def filter_dataset(annotations_dict, visceral_slides, positive_patients):

    positive_visceral_slides = []
    negative_visceral_slides = []
    for visceral_slide in visceral_slides:
        if (
            visceral_slide.patient_id in positive_patients
            and visceral_slide.full_id in annotations_dict
        ):
            positive_visceral_slides.append(visceral_slide)
        else:
            negative_visceral_slides.append(visceral_slide)

    # np.random.shuffle(negative_visceral_slides)
    negative_visceral_slides = negative_visceral_slides[: len(positive_visceral_slides)]

    return np.concatenate((positive_visceral_slides, negative_visceral_slides))


class DetectionConfig:
    def __init__(self):
        self.cumulative_vs = True
        self.anterior_motion_norm = True
        self.norm_by_exp = False
        self.vs_stat = False
        self.kfold = False
        self.expectation_norm_type = VSExpectationNormType.mean_div
        self.transform = VSTransform.none
        self.negative_vs_needed = (
            self.norm_by_exp
            and self.expectation_norm_type == VSExpectationNormType.standardize
        )
        self.vis_prior = True  # not anterior_motion_norm
        self.adhesion_types = [AdhesionType.anteriorWall, AdhesionType.pelvis]
        # self.adhesion_types = [AdhesionType.pelvis]
        self.conf_type = ConfidenceType.mean
        self.region_growing_ind = 2.5
        self.min_region_len = 5
        self.vis_vs = False
        self.exp_prefix = ""
        self.lr_suffix = ""
        self.exp_name = ""


def get_experiment_path(config, output_path):

    if len(config.adhesion_types) > 1:
        root_folder = "all_data_experiments"
    elif config.adhesion_types[0] == AdhesionType.pelvis:
        root_folder = "pelvis_experiments"
    else:
        root_folder = "anterior_wall_experiments"

    normalization_folder = (
        "anterior_motion_norm" if config.anterior_motion_norm else "vicinity_norm"
    )

    vs_type_folder = "cumulative" if config.cumulative_vs else "inspexp"
    if config.norm_by_exp:
        vs_type_folder += (
            "_mean_div"
            if config.expectation_norm_type == VSExpectationNormType.mean_div
            else "_stand"
        )
        if config.transform == VSTransform.sqrt:
            vs_type_folder += "_sqrt"
        elif config.transform == VSTransform.log:
            vs_type_folder += "_log"

    experiment_folder = (
        "rgi{}".format(config.region_growing_ind)
        if config.region_growing_ind is not None
        else "nonlin"
    )
    experiment_folder += "_mrl{}".format(config.min_region_len)
    if config.kfold:
        experiment_folder += "_conf_lr" + config.lr_suffix
    else:
        experiment_folder += (
            "_conf_mean" if config.conf_type == ConfidenceType.mean else "_conf_min"
        )

    if len(config.exp_prefix) > 0:
        experiment_folder = config.exp_prefix + "_" + experiment_folder

    experiment_path = (
        output_path
        / root_folder
        / normalization_folder
        / vs_type_folder
        / experiment_folder
    )
    return experiment_path


# Cumulative VS, normalised by division at mean with applied sqrt transformation
def run_detection_pipeline_test(detection_path, output_path):

    experiment_path = output_path / "test_eval_lr"

    # Necessary train data
    # train_images_path = detection_path / IMAGES_FOLDER / TRAIN_FOLDER
    vs_path_train = detection_path / VS_FOLDER / AVG_NORM_FOLDER / CUMULATIVE_VS_FOLDER
    annotations_train_path = (
        detection_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    )
    annotations_dict_train = load_annotations(
        annotations_train_path,
        as_dict=True,
        adhesion_types=[AdhesionType.anteriorWall, AdhesionType.pelvis],
    )
    visceral_slides_train = load_visceral_slides(vs_path_train)

    vs_expectation_path = (
        detection_path / METADATA_FOLDER / CUMULATIVE_VS_EXPECTATION_FILE_SQRT
    )
    with open(vs_expectation_path, "r+b") as file:
        vs_expectation_dict = pickle.load(file)
        expectation = vs_expectation_dict["means"], vs_expectation_dict["stds"]
        means = expectation[0]
        stds = expectation[1]

    # Normalise train vs
    for vs in visceral_slides_train:
        vs.values = np.sqrt(vs.values)
        vs.norm_with_expectation(means, stds, VSExpectationNormType.mean_div)

    # Necessary test data
    test_images_path = detection_path / IMAGES_FOLDER / TEST_FOLDER
    vs_path_test = (
        detection_path / VS_TEST_FOLDER / AVG_NORM_FOLDER / CUMULATIVE_VS_FOLDER
    )
    annotations_test_path = (
        detection_path / METADATA_FOLDER / BB_TEST_ANNOTATIONS_EXPANDED_FILE
    )
    annotations_dict_test = load_annotations(
        annotations_test_path,
        as_dict=True,
        adhesion_types=[AdhesionType.anteriorWall, AdhesionType.pelvis],
    )
    visceral_slides_test = load_visceral_slides(vs_path_test)

    # Normalise test vs
    for vs in visceral_slides_test:
        vs.values = np.sqrt(vs.values)
        vs.norm_with_expectation(means, stds, VSExpectationNormType.mean_div)

    # Vis test VS
    # vis_path = experiment_path / "vis_vs"
    # vis_computed_cum_vs(visceral_slides_test, test_images_path, vis_path)

    # plot_vs_distribution(visceral_slides_test, experiment_path)
    # plot_vs_distribution(visceral_slides_test, experiment_path, prior_only=True)
    # vs_adhesion_boxplot(visceral_slides_test, annotations_test_path, experiment_path)
    # vs_adhesion_boxplot(visceral_slides_test, annotations_test_path, experiment_path, prior_only=True)

    # Train LR at training set
    lr = trainLR(visceral_slides_train, annotations_dict_train)

    # Predict
    predictions = predict(
        visceral_slides_test,
        annotations_dict_test,
        False,
        ConfidenceType.mean,
        2.5,
        5,
        lr,
    )

    # Evaluate
    evaluate(predictions, annotations_dict_test, experiment_path)

    # Visualise
    visualize(
        visceral_slides_test,
        annotations_dict_test,
        predictions,
        test_images_path,
        experiment_path,
        True,
    )


def run_detection_pipeline(config, detection_path, output_path):
    images_path = detection_path / IMAGES_FOLDER / TRAIN_FOLDER
    vs_folder_root = (
        AVG_NORM_FOLDER if config.anterior_motion_norm else VICINITY_NORM_FOLDER
    )
    vs_folder = CUMULATIVE_VS_FOLDER if config.cumulative_vs else INS_EXP_VS_FOLDER
    experiment_path = get_experiment_path(config, output_path)

    if config.anterior_motion_norm:
        if config.cumulative_vs:
            control_stat_file = (
                CUMULATIVE_VS_EXPECTATION_FILE_SQRT
                if config.transform == VSTransform.sqrt
                else CUMULATIVE_VS_EXPECTATION_FILE
            )
        else:
            control_stat_file = (
                INSPEXP_VS_EXPECTATION_FILE_SQRT
                if config.transform == VSTransform.sqrt
                else INSPEXP_VS_EXPECTATION_FILE
            )
    else:
        if config.cumulative_vs:
            control_stat_file = (
                CUMULATIVE_VS_EXPECTATION_VICINITY_FILE_SQRT
                if config.transform == VSTransform.sqrt
                else CUMULATIVE_VS_EXPECTATION_VICINITY_FILE
            )
        else:
            control_stat_file = (
                INSPEXP_VS_EXPECTATION_VICINITY_FILE_SQRT
                if config.transform == VSTransform.sqrt
                else INSPEXP_VS_EXPECTATION_VICINITY_FILE
            )

    vs_path = detection_path / VS_FOLDER / vs_folder_root / vs_folder
    vs_expectation_path = detection_path / METADATA_FOLDER / control_stat_file
    with open(vs_expectation_path, "r+b") as file:
        vs_expectation_dict = pickle.load(file)
        expectation = vs_expectation_dict["means"], vs_expectation_dict["stds"]

    inspexp_file_path = detection_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    annotations_path = detection_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    patients_split_file = detection_path / METADATA_FOLDER / PATIENTS_SPLIT_FILE_NAME
    with open(patients_split_file) as f:
        patients_split = json.load(f)
        positive_patients = patients_split["train"]["positive"]

    annotations_dict = load_annotations(
        annotations_path, as_dict=True, adhesion_types=config.adhesion_types
    )
    visceral_slides = load_visceral_slides(vs_path)
    # visceral_slides.sort(key=lambda x: x.full_id)

    # Filter out only when one type is considered
    if len(config.adhesion_types) == 1:
        visceral_slides = filter_dataset(
            annotations_dict, visceral_slides, positive_patients
        )

    # Normalize by expectation
    if config.norm_by_exp:
        means = expectation[0]
        stds = expectation[1]
        for vs in visceral_slides:
            if config.transform == VSTransform.log:
                vs.values = np.log(vs.values)
            elif config.transform == VSTransform.sqrt:
                vs.values = np.sqrt(vs.values)

            vs.norm_with_expectation(means, stds, config.expectation_norm_type)

    # vs_min_stat(visceral_slides, annotations_dict)

    if config.vis_vs:
        vis_path = experiment_path / "vis_vs"
        if config.cumulative_vs:
            vis_computed_cum_vs(visceral_slides, images_path, vis_path)
        else:
            vis_computed_cum_vs(visceral_slides, images_path, vis_path, inspexp_data)

    if config.vs_stat:
        plot_vs_distribution(visceral_slides, experiment_path)
        plot_vs_distribution(visceral_slides, experiment_path, prior_only=True)
        vs_adhesion_boxplot(visceral_slides, annotations_path, experiment_path)
        vs_adhesion_boxplot(
            visceral_slides, annotations_path, experiment_path, prior_only=True
        )

    # load 5-fold cv split, filter out and run
    if config.kfold:
        detection_slices_kfold_file = (
            detection_path / METADATA_FOLDER / DETECTION_SLICE_FOLD_FILE_NAME
        )
        with open(detection_slices_kfold_file) as f:
            kfolds = json.load(f)

        all_predictions = {}
        for fold in kfolds:
            train_ids = fold["train"]
            val_ids = fold["val"]

            train_visceral_slides = [
                vs for vs in visceral_slides if vs.full_id in train_ids
            ]
            val_visceral_slides = [
                vs for vs in visceral_slides if vs.full_id in val_ids
            ]

            lr = trainLR(train_visceral_slides, annotations_dict)
            predictions = predict(
                val_visceral_slides,
                annotations_dict,
                config.negative_vs_needed,
                config.conf_type,
                config.region_growing_ind,
                config.min_region_len,
                lr,
            )
            all_predictions.update(predictions)

        evaluate(all_predictions, annotations_dict, experiment_path)
        if config.cumulative_vs:
            visualize(
                visceral_slides,
                annotations_dict,
                all_predictions,
                images_path,
                experiment_path,
                config.vis_prior,
            )
        else:
            visualize(
                visceral_slides,
                annotations_dict,
                all_predictions,
                images_path,
                experiment_path,
                config.vis_prior,
                inspexp_data,
            )
    else:
        predictions = predict(
            visceral_slides,
            annotations_dict,
            config.negative_vs_needed,
            config.conf_type,
            config.region_growing_ind,
            config.min_region_len,
        )
        evaluate(predictions, annotations_dict, experiment_path)
        if config.cumulative_vs:
            visualize(
                visceral_slides,
                annotations_dict,
                predictions,
                images_path,
                experiment_path,
                config.vis_prior,
            )
        else:
            visualize(
                visceral_slides,
                annotations_dict,
                predictions,
                images_path,
                experiment_path,
                config.vis_prior,
                inspexp_data,
            )


def vs_min_stat(visceral_slides, annotations_dict):
    negative_mins = []
    positive_mins = []

    for vs in visceral_slides:
        x, y, slide_value = vs.x, vs.y, vs.values
        vs_region = filter_out_prior_vs_subset(x, y, slide_value)
        vs_region_min = np.min(vs_region[:, 2])
        if vs.full_id in annotations_dict:
            negative_mins.append(vs_region_min)
        else:
            positive_mins.append(vs_region_min)

    print("Mean min positive VS {}".format(np.mean(negative_mins)))
    print("Mean min negative VS {}".format(np.mean(positive_mins)))

    t_test = stats.ttest_ind(negative_mins, positive_mins, equal_var=True)
    print(
        "Equal variances T-stat {}, p-value {}".format(t_test.statistic, t_test.pvalue)
    )

    t_test = stats.ttest_ind(negative_mins, positive_mins, equal_var=False)
    print(
        "Non equal variances T-stat {}, p-value {}".format(
            t_test.statistic, t_test.pvalue
        )
    )


def compare_experiments(
    experiments_paths,
    experiment_names,
    colors,
    output_path,
    metrics_to_plot=[
        EvaluationMetrics.froc_all,
        EvaluationMetrics.precision,
        EvaluationMetrics.recall,
        EvaluationMetrics.slice_roc,
    ],
):

    output_path.mkdir(exist_ok=True, parents=True)

    frocs_all = []
    frocs_neg = []
    precisions = []
    recalls = []
    slice_rocs = []

    for metrics_path in experiments_paths:
        froc, pr_curve, slice_roc, _, _ = load_evaluation_metrics(metrics_path)

        frocs_all.append((froc[0], froc[2]))
        frocs_neg.append((froc[1], froc[2]))
        precisions.append((pr_curve[0], pr_curve[2]))
        recalls.append((pr_curve[1], pr_curve[2]))
        slice_rocs.append(slice_roc)

    if EvaluationMetrics.froc_all in metrics_to_plot:
        plot_FROCs(frocs_all, output_path, "Frocs all", experiment_names, colors)

    if EvaluationMetrics.froc_negative in metrics_to_plot:
        plot_FROCs(frocs_neg, output_path, "Frocs negative", experiment_names, colors)

    if EvaluationMetrics.precision in metrics_to_plot:
        plot_precisions_recalls(
            precisions, output_path, legends=experiment_names, colors=colors
        )

    if EvaluationMetrics.recall in metrics_to_plot:
        plot_precisions_recalls(
            recalls,
            output_path,
            metric="Recall",
            legends=experiment_names,
            colors=colors,
        )

    if EvaluationMetrics.slice_roc in metrics_to_plot:
        plot_ROCs(
            slice_rocs, output_path=output_path, legends=experiment_names, colors=colors
        )


def compare_experiments_conf(
    configs,
    colors,
    experiments_path,
    output_path,
    metrics_to_plot=[
        EvaluationMetrics.froc_all,
        EvaluationMetrics.precision,
        EvaluationMetrics.recall,
        EvaluationMetrics.slice_roc,
    ],
):

    experiments_paths = []
    experiment_names = []

    for config in configs:
        metrics_path = (
            get_experiment_path(config, experiments_path) / EVALUATION_METRICS_FILE
        )
        experiments_paths.append(metrics_path)
        experiment_names.append(config.exp_name)

    compare_experiments(
        experiments_paths, experiment_names, colors, output_path, metrics_to_plot
    )


def test_vs_calc_loading():
    np.random.seed(99)

    detection_path = Path(DETECTION_PATH)
    experiments_path = Path(DETECTION_PATH) / "experiments_thesis"

    colors_cum = cm.get_cmap("Blues")(np.linspace(0.3, 0.9, 3))
    colors_inspexp = cm.get_cmap("Oranges")(np.linspace(0.3, 0.9, 3))
    colors = np.concatenate((colors_cum, colors_inspexp))

    # run_detection_pipeline_test(detection_path, experiments_path)

    print("\nMean")
    config1 = DetectionConfig()
    config1.cumulative_vs = True
    config1.vis_vs = False
    config1.kfold = True
    config1.conf_type = ConfidenceType.mean
    config1.norm_by_exp = True
    config1.expectation_norm_type = VSExpectationNormType.mean_div
    config1.transform = VSTransform.sqrt
    config1.exp_name = "conf_mean"
    # config1.region_growing_ind = 5
    # config1.min_region_len = 3
    run_detection_pipeline(config1, detection_path, experiments_path)

    """
    ex1 = experiments_path / "test_eval_min"  / EVALUATION_METRICS_FILE
    ex2 = experiments_path / "test_eval_mean" / EVALUATION_METRICS_FILE
    ex3 = experiments_path / "test_eval_lr" / EVALUATION_METRICS_FILE

    compare_experiments([ex1, ex2, ex3], ["conf_min", "conf_mean", "conf_lr"], colors_cum,
                        experiments_path / "comparison_test" / "cum_mean_div_sqrt")
    """

    """
    ex1 = experiments_path / "pelvis_experiments" / "anterior_motion_norm" / "cumulative" / "rgi2.5_mrl5_conf_mean" / EVALUATION_METRICS_FILE
    ex2 = experiments_path / "pelvis_experiments" / "anterior_motion_norm" / "cumulative_mean_div_sqrt" / "rgi2.5_mrl5_conf_lr" / EVALUATION_METRICS_FILE
    ex3 = experiments_path / "pelvis_experiments" / "anterior_motion_norm" / "cumulative_stand_sqrt" / "rgi5_mrl5_conf_min" / EVALUATION_METRICS_FILE

    #ex4 = experiments_path / "all_data_experiments" / "anterior_motion_norm" / "inspexp_stand_sqrt" / "rgi7_mrl3_conf_min" / EVALUATION_METRICS_FILE
    #ex5 = experiments_path / "all_data_experiments" / "anterior_motion_norm" / "inspexp_stand_sqrt" / "rgi7_mrl3_conf_mean" / EVALUATION_METRICS_FILE
    #ex6 = experiments_path / "all_data_experiments" / "anterior_motion_norm" / "inspexp_stand_sqrt" / "rgi7_mrl3_conf_lr" / EVALUATION_METRICS_FILE

    compare_experiments([ex1, ex2, ex3], ["unnrom", "div_mean", "stand"], colors_cum,
                        experiments_path / "comparison_pelvis" / "cumulative")

    #compare_experiments([ex4, ex5, ex6], ["conf_min", "conf_mean", "conf_lr"], colors_inspexp,
    #                    experiments_path / "comparison" / "inspexp_stand_sqrt")

    #compare_experiments([ex1, ex2, ex3, ex4, ex5, ex6], ["cum_unnrom", "cum_div_mean", "cum_stand", "inspexp_unnrom", "inspexp_div_mean", "inspexp_stand"], colors, experiments_path / "comparison" / "all_lr")
    """
    """
    print("Min")
    config = DetectionConfig()
    #config.cumulative_vs = False
    config.vs_stat = False
    config.conf_type = ConfidenceType.min
   # config.norm_by_exp = True
   # config.expectation_norm_type = VSExpectationNormType.standardize
    #config.transform = VSTransform.sqrt
    config.exp_name = "conf_min"
    #config.region_growing_ind = 5
    #config.min_region_len = 3
    run_detection_pipeline(config, detection_path, experiments_path)

    print("\nMean")
    config1 = DetectionConfig()
    #config1.cumulative_vs = False
    config1.vs_stat = False
    config1.conf_type = ConfidenceType.mean
    #config1.norm_by_exp = True
    #config1.expectation_norm_type = VSExpectationNormType.standardize
    #config1.transform = VSTransform.sqrt
    config1.exp_name = "conf_mean"
    #config1.region_growing_ind = 5
    #config1.min_region_len = 3
    run_detection_pipeline(config1, detection_path, experiments_path)

    print("\nLR")
    config2 = DetectionConfig()
    #config2.cumulative_vs = False
    config2.vs_stat = False
    config2.kfold = True
    #config2.norm_by_exp = True
    #config2.expectation_norm_type = VSExpectationNormType.standardize
    #config2.transform = VSTransform.sqrt
    config2.exp_name = "conf_lr"
    #config2.region_growing_ind = 5
    #config2.min_region_len = 3
    run_detection_pipeline(config2, detection_path, experiments_path)

    output_path = experiments_path / "comparison_ant_wall" / "cum_unnorm"
    compare_experiments_conf([config, config1, config2], colors_cum, experiments_path, output_path)

    """

    """

    config1 = DetectionConfig()
    config1.conf_type = ConfidenceType.min
    config1.norm_by_exp = True
    config1.exp_name = "min"
    config1.transform = VSTransform.sqrt

    run_detection_pipeline(config1, detection_path, experiments_path)

    config2 = DetectionConfig()
    config2.kfold = True
    config2.norm_by_exp = True
    config2.exp_name = "lr"
    config2.transform = VSTransform.sqrt

    run_detection_pipeline(config2, detection_path, experiments_path)

    output_path = experiments_path / "comparison" / "pelvis_ant_motion_norm_cum_mean_div_sqrt"
    compare_experiments([config, config1, config2], experiments_path, output_path)


    config = DetectionConfig()
    config.conf_type = ConfidenceType.mean
    config.exp_name = "unnrom_mean"

    config1 = DetectionConfig()
    config1.conf_type = ConfidenceType.mean
    config1.norm_by_exp = True
    config1.exp_name = "mean_div_mean"

    config2 = DetectionConfig()
    config2.conf_type = ConfidenceType.mean
    config2.norm_by_exp = True
    config2.exp_name = "mean_div_mean_sqrt"
    config2.transform = VSTransform.sqrt

    output_path = experiments_path / "comparison" / "pelvis_ant_motion_norm_cum"
    compare_experiments([config, config1, config2], experiments_path, output_path)

    
    config1 = DetectionConfig()
    config1.kfold = True
    config1.norm_by_exp = True
    config1.exp_name = "cum_mean_div"

    config2 = DetectionConfig()
    config2.kfold = True
    config2.norm_by_exp = True
    config2.transform = VSTransform.sqrt
    config2.exp_name = "cum_mean_div_sqrt"

    config3 = DetectionConfig()
    config3.cumulative_vs = False
    config3.norm_by_exp = True
    config3.min_region_len = 3
    config3.region_growing_ind = 7
    config3.kfold = True
    config3.transform = VSTransform.sqrt
    config3.exp_name = "insexp_mean_div_sqrt"

    config4 = DetectionConfig()
    config4.cumulative_vs = False
    config4.norm_by_exp = True
    config4.conf_type = ConfidenceType.min
    config4.expectation_norm_type = VSExpectationNormType.standardize
    config4.min_region_len = 5
    config4.region_growing_ind = 5
    config4.exp_name = "insexp_stand"

    
    config5 = DetectionConfig()
    config5.cumulative_vs = False
    config5.norm_by_exp = True
    config5.kfold = True
    config5.expectation_norm_type = VSExpectationNormType.standardize
    config5.transform = VSTransform.sqrt
    config5.exp_name = "stand_sqrt"
    config5.min_region_len = 3
    config5.region_growing_ind = 7

    output_path = experiments_path / "comparison" / "ant_motion_norm"
    compare_experiments([config1, config2, config3, config4], experiments_path, output_path)
    """


if __name__ == "__main__":
    test_vs_calc_loading()
