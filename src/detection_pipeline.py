import numpy as np
import SimpleITK as sitk
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from adhesions import AdhesionType, Adhesion, load_annotations
from config import *
from cinemri.config import ARCHIVE_PATH
from cinemri.definitions import CineMRISlice
from utils import bb_size_stat, load_visceral_slides, binning_intervals, get_inspexp_frames, adhesions_stat
from contour import get_connected_regions, get_adhesions_prior_coords, get_abdominal_contour_top
from froc.deploy_FROC import y_to_FROC
from scipy import stats
from enum import Enum, unique
from sklearn.metrics import roc_curve, auc
from vs_definitions import VSExpectationNormType

VISCERAL_SLIDE_PATH = "../../data/visceral_slide_all/visceral_slide"
PREDICTION_COLOR = (0, 0.8, 0.2)

# TODO: rename
def extract_visceral_slide_dict(annotations, visceral_slide_path):
    visceral_slide_dict = {}
    for annotation in annotations:
        visceral_slide_results_path = annotation.build_path(visceral_slide_path, extension="")
        if visceral_slide_results_path.exists():
            # Load the computed visceral slide
            visceral_slide_file_path = visceral_slide_results_path / VISCERAL_SLIDE_FILE
            with open(str(visceral_slide_file_path), "r+b") as file:
                visceral_slide_data = pickle.load(file)
                visceral_slide_dict[annotation.full_id] = visceral_slide_data

    return visceral_slide_dict


def bb_from_region(region, mean_size):
    """
    Calculates bounding box based on the coordinates of the points in the region
    and mean annotations width and height
    
    Parameters
    ----------
    region : list of list
       A list of contour points represented as 2d arrays

    mean_size : tuple of float
       Mean size of adhesions bounding boxes in annotations. The first values is width and the second is height

    Returns
    -------
    adhesion : Adhesion
       A bounding box corresponding to the region
    """

    # axis=0: x, axis=1: y
    def compute_origin_and_len(region, mean_size, axis=0):
        region_centre = region[round(len(region) / 2) - 1]

        axis_min = np.min([coord[axis] for coord in region])
        axis_max = np.max([coord[axis] for coord in region])
        length = axis_max - axis_min + 1

        if length < mean_size[axis]:
            origin = region_centre[axis] - round(mean_size[axis] / 2)
            length = mean_size[axis]
        else:
            origin = axis_min

        return origin, length

    origin_x, width = compute_origin_and_len(region, mean_size, axis=0)
    origin_y, height = compute_origin_and_len(region, mean_size, axis=1)
    return Adhesion([origin_x, origin_y, width, height])


def region_size(region):

    def compute_len(region, axis=0):
        axis_min = np.min([coord[axis] for coord in region])
        axis_max = np.max([coord[axis] for coord in region])
        length = axis_max - axis_min + 1
        return length

    width = compute_len(np.array(region), axis=0)
    height = compute_len(np.array(region), axis=1)
    return width, height


def bb_adjusted_region(region, bb):
    x_inside_bb = (bb.origin_x <= region[:, 0]) & (region[:, 0] <= bb.max_x)
    y_inside_bb = (bb.origin_y <= region[:, 1]) & (region[:, 1] <= bb.max_y)
    inside_bb = x_inside_bb & y_inside_bb
    adjusted_region = region[inside_bb]

    adjusted_region_start = adjusted_region[0, :2]
    adjusted_region_end = adjusted_region[-1, :2]

    coords = region[:, :2]
    start_index = np.where((coords == adjusted_region_start).all(axis=1))[0][0]
    end_index = np.where((coords == adjusted_region_end).all(axis=1))[0][0]

    if start_index > end_index:
        temp = start_index
        start_index = end_index
        end_index = temp

    return adjusted_region, start_index, end_index


def find_min_vs(regions):

    min_region_ind = None
    min_slide_value = np.inf
    # Find minimum across region
    for (region_ind, region) in enumerate(regions):
        region_values = region[:, 2]
        region_min = np.min(region_values)
        if region_min < min_slide_value:
            min_region_ind = region_ind
            min_slide_value = region_min

    region_of_prediction = regions[min_region_ind]
    vs_values = region_of_prediction[:, 2]
    vs_value_min_ind = np.argmin(vs_values)

    return region_of_prediction, min_region_ind, vs_values, vs_value_min_ind


def adhesions_from_region_fixed_size(regions, bb_size, bb_size_max, vs_max):

    # Predict bounding boxes
    region_len = max(bb_size[0], bb_size[1])
    vicinity = round((region_len - 1) / 2)
    bounding_boxes = []
    # While there are regions that are larger than region_len
    while len(regions) > 0:
        region_of_prediction, min_region_ind, vs_values, vs_value_min_ind = find_min_vs(regions)
        region_start = max(0, vs_value_min_ind - vicinity)
        region_end = min(len(vs_values), vs_value_min_ind + vicinity)

        # Generate bounding box from region
        bb_region = region_of_prediction[region_start:region_end]
        bounding_box = bb_from_region(bb_region, bb_size)

        adjusted_region, start_index, end_index = bb_adjusted_region(region_of_prediction, bounding_box)

        # Later invert - subtract confidences from the max VS value
        # Take mean region vs as confidence
        confidence = vs_max - np.mean(adjusted_region[:, 2])
        bounding_boxes.append([bounding_box, confidence])

        # Cut out bounding box region
        # Remove the region of prediction from the array
        del regions[min_region_ind]
        # Add regions around the cutoff if they are large enough
        region_before_bb = region_of_prediction[:start_index]
        if len(region_before_bb) > region_len:
            regions.append(region_before_bb)

        region_after_bb = region_of_prediction[end_index:]
        if len(region_after_bb) > region_len:
            regions.append(region_after_bb)

    return bounding_boxes


def adhesions_with_region_growing(regions, bb_size, bb_size_max, vs_max, vicinity_ind=2.5, region_growing_ind=2.5, min_region_len=5):
    region_len = max(bb_size[0], bb_size[1])
    vicinity = region_len - 1
    vicinity_max = vicinity_ind * vicinity

    bounding_boxes = []
    # While there are regions that are larger than  min_region_len
    while len(regions) > 0:
        region_of_prediction, min_region_ind, vs_values, vs_value_min_ind = find_min_vs(regions)
        min_slide_value = vs_values[vs_value_min_ind]
        max_region_slide_value = min_slide_value * region_growing_ind if min_slide_value > 0 else min_slide_value / region_growing_ind

        # Looking for the regions boundaries
        start_ind = end_ind = vs_value_min_ind
        start_ind_found = end_ind_found = False
        prediction_region = [region_of_prediction[vs_value_min_ind]]
        while not (start_ind_found and end_ind_found):
            new_start_ind = max(0, start_ind - 1)
            start_value = vs_values[start_ind]

            exceeded_size_limit = region_size(prediction_region)[0] >= bb_size_max[0] or region_size(prediction_region)[1] >= bb_size_max[1]
            start_ind_found = start_value > max_region_slide_value or exceeded_size_limit or start_ind == 0
            if not start_ind_found:
                start_ind = new_start_ind
                prediction_region.append(region_of_prediction[start_ind])

            new_end_ind = min(len(region_of_prediction) - 1, end_ind + 1)
            end_value = vs_values[end_ind]

            exceeded_size_limit = region_size(prediction_region)[0] >= bb_size_max[0] or region_size(prediction_region)[
                1] >= bb_size_max[1]
            end_ind_found = end_value > max_region_slide_value or exceeded_size_limit or end_ind == (len(region_of_prediction) - 1)
            if not end_ind_found:
                end_ind = new_end_ind
                prediction_region.append(region_of_prediction[end_ind])

            if new_start_ind == 0 and new_end_ind == (len(region_of_prediction) - 1):
                start_ind_found = end_ind_found = True

        # Only predict the bouding box if the region is large enough
        if end_ind - start_ind >= min_region_len:
            # Generate bounding box from region
            bb_region = region_of_prediction[start_ind:end_ind]
            bounding_box = bb_from_region(bb_region, bb_size)

            adjusted_region, start_ind, end_ind = bb_adjusted_region(region_of_prediction, bounding_box)
            # Take mean region vs as confidence
            confidence = vs_max - min_slide_value
            bounding_boxes.append([bounding_box, confidence])

        # Cut out bounding box region
        # Remove the region of prediction from the array
        del regions[min_region_ind]
        # Add regions around the cutoff if they are large enough
        region_before_bb = region_of_prediction[:start_ind]
        if len(region_before_bb) > region_len:
            regions.append(region_before_bb)

        region_after_bb = region_of_prediction[end_ind:]
        if len(region_after_bb) > region_len:
            regions.append(region_after_bb)

    return bounding_boxes


def bb_with_threshold(vs, bb_size, bb_size_max, vs_range, pred_func):
    """
    Predicts adhesions with bounding boxes based on the values of the normalized visceral slide and the
    specified threshold level

    Parameters
    ----------
    x, y : ndarray of int
       x-axis and y-axis components of abdominal cavity contour
    slide_value : ndarray of float
       Normalized values of visceral slide corresponding to the coordinates of abdominal cavity contour
    bb_size : tuple of float
       Mean size of adhesions bounding boxes in annotations. The first values is width and the second is height
    Returns
    -------
    bounding_boxes : list of Adhesion
       A list of bounding boxes predicted based on visceral slide values
    """


    x, y, slide_value = vs.x, vs.y, vs.values

    # Filter out the region in which no adhesions can be present
    x_prior, y_prior = get_adhesions_prior_coords(x, y)

    coords = np.column_stack((x, y)).tolist()
    prior_coords = np.column_stack((x_prior, y_prior)).tolist()
    prior_inds = [ind for ind, coord in enumerate(coords) if coord in prior_coords]

    contour_subset = np.column_stack((x, y, slide_value))
    contour_subset = contour_subset[prior_inds]

    contour_subset = np.array([vs for vs in contour_subset if vs_range[0] <= vs[2] <= vs_range[1]])
    if len(contour_subset) == 0:
        return []

    # Predict
    suitable_regions = get_connected_regions(contour_subset)
    bounding_boxes = pred_func(suitable_regions, bb_size, bb_size_max, vs_range[1])
    return bounding_boxes


# TODO: refactor or move to a more appropriate place
def adhesion_likelihood(region, vs_min, vs_max, vs_threshold):
    # Average

    """
    region_slides = [comp[2] for comp in region]
    # or median?
    mean_region_slide = np.median(region_slides)
    likelihood = (vs_threshold - mean_region_slide) / (vs_threshold - vs_min)
    """

    # Region likelihood

    likelihood = 1
    for point in region:
        likelihood *= (vs_max - point[2]) / (vs_max - vs_min)

    return likelihood


def visualize_gt_vs_prediction(prediction, annotation, x, y, slide_normalized, insp_frame, file_path=None):
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
    slide_normalized : ndarray of float
       Normalized values of visceral slide corresponding to the coordinates of abdominal cavity contour
    insp_frame : 2d ndarray
       An inspiration frame
    file_path : Path
       A Path where to save a file, default is None
    """

    plt.figure()
    plt.imshow(insp_frame, cmap="gray")
    # Plot viseral slide
    plt.scatter(x, y, s=5, c=slide_normalized, cmap="jet")
    plt.colorbar()
    # Plot regions of slow slide
    # plt.scatter(low_slide_x, low_slide_y, s=10, color="k", marker="x")
    ax = plt.gca()

    for adhesion, confidence in prediction:
        adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                  linewidth=1.5, edgecolor=PREDICTION_COLOR, facecolor='none')
        ax.add_patch(adhesion_rect)
        plt.text(adhesion.origin_x, adhesion.origin_y - 3, "{:.3f}".format(confidence), c=PREDICTION_COLOR, fontweight='semibold')

    if annotation:
        for adhesion in annotation.adhesions:
            adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(adhesion_rect)

    plt.axis("off")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()

# TODO: probably leave only prior coords when plotting
def predict_and_visualize(visceral_slides, annotations_dict, images_path, output_path, bb_size_median=False,
                          vs_expectation=None, expectation_norm_type=VSExpectationNormType.mean_div, inspexp_data=None):

    output_path.mkdir(exist_ok=True, parents=True)

    annotations = annotations_dict.values()
    # Average bounding box size
    bb_size = bb_size_stat(annotations, is_median=bb_size_median)

    # Normalize with expectation
    if vs_expectation:
        for vs in visceral_slides:
            vs.norm_with_expectation(vs_expectation[0], vs_expectation[1], expectation_norm_type)

    # vary threshold level
    # Get predictions by visceral slide level threshold
    # If we normalize by motion by standardization, we are interested only in the negative range
    negative_vs_needed = vs_expectation is not None and expectation_norm_type == VSExpectationNormType.standardize
    vs_min, vs_max = (-np.inf, 0) if negative_vs_needed else get_vs_range(visceral_slides)

    for vs in visceral_slides:
        prediction = bb_with_threshold(vs, bb_size, (vs_min, vs_max), adhesions_with_region_growing)

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
                print("Missing insp/exp data for the patient {}, study {}, slice {}".format(vs.patient_id,
                                                                                            vs.study_id,
                                                                                            vs.slice_id))

        file_path = output_path / (vs.full_id + ".png")
        annotation = annotations_dict[vs.full_id] if vs.full_id in annotations_dict else None
        visualize_gt_vs_prediction(prediction, annotation, vs.x, vs.y, vs.values, frame, file_path)


# TODO: should we also output confidence here?
def predict(visceral_slides, bb_size, bb_size_max, vs_range):
    """
    Predicts adhesions based on visceral slide values for the provided full ids of slices

    Parameters
    ----------
    full_ids : list of str
       Full ids of slices in the format patientID_studyID_slice_ID
    visceral_slides : dict
       A dictionary that contains the coordinates of abdominal cavity contour and
       the corresponding visceral slide
    bb_size : tuple of float
       Mean size of adhesions bounding boxes in annotations. The first values is width and the second is height

    Returns
    -------
    prediction : dict
       A dictionary with full ids of slices used as keys and predicted bounding boxes as values
    """

    predictions = {}

    for vs in visceral_slides:
        prediction = bb_with_threshold(vs, bb_size, bb_size_max, vs_range, adhesions_with_region_growing)
        predictions[vs.full_id] = prediction

    return predictions

# Some annotations are very small - setup min annotation size
# Min width and height - 15 +

# TODO: understand adequate evaluation metric
# Recall and precision since we do not have confidence score?
# F1 score
# Maybe confidence - inverse of length? Ask supervisors about it
def evaluate(annotations, predictions, iou_threshold=0.1, visceral_slide_dict=None, images_path=None, inspexp_data=None):

    tps = []
    tp_num = 0
    fps = []
    fns = []
    prediction_num = 0
    for annotation in annotations:
        prediction = predictions[annotation.full_id]
        # Tracks which predictions has been assigned to a TP
        hit_predictions_inds = []
        tp_num += len(annotation.adhesions)
        prediction_num += len(prediction)
        # Loop over TPs
        for adhesion in annotation.adhesions:
            adhesion.adjust_size()

            max_iou = 0
            max_iou_ind = -1
            # For simplicity for now one predicted bb can correspond to only one TP
            for ind, bounding_box in enumerate(prediction):
                curr_iou = adhesion.iou(bounding_box)
                # Do not use predictions that have already been assigned to a TP
                if curr_iou > max_iou and ind not in hit_predictions_inds:
                    max_iou = curr_iou
                    max_iou_ind = ind

            # If a maximum IoU is greater than the threshold, consider a TP as found
            if max_iou >= iou_threshold:
                matching_pred = prediction[max_iou_ind]
                tps.append((adhesion, matching_pred))
                hit_predictions_inds.append(max_iou_ind)
            # Otherwise add it as a false negative
            else:
                fns.append(adhesion)

        # Get false positive
        # Predictions that was not matched with a TP are FPs
        tp_mask = np.full(len(prediction), True, dtype=bool)
        tp_mask[hit_predictions_inds] = False
        fps_left = np.array(prediction)[tp_mask]
        fps.append(fps_left)

        # Optionally visualise
        if visceral_slide_dict is not None:
            visceral_slide_data = visceral_slide_dict[annotation.full_id]
            x, y = visceral_slide_data["x"], visceral_slide_data["y"]
            visceral_slide = visceral_slide_data["slide"]
            slide_normalized = np.abs(visceral_slide) / np.abs(visceral_slide).max()

            # Extract inspiration frame
            try:
                patient_data = inspexp_data[annotation.patient_id]
                study_data = patient_data[annotation.study_id]
                inspexp_frames = study_data[annotation.slice_id]
            except:
                print("Missing insp/exp data for the patient {}, study {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.study_id,
                                                                                           annotation.slice_id))

            # load images
            slice_path = annotation.build_path(images_path)
            slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
            insp_frame = slice_array[inspexp_frames[0]].astype(np.uint32)

            visualize_gt_vs_prediction(annotation, prediction, x, y, slide_normalized, insp_frame)
            print("vis")


    # How many of predicted bbs are TP
    precision = len(tps) / prediction_num
    print("Precision {}".format(precision))

    # How many of TP are found
    recall = len(tps) / tp_num
    print("Recall {}".format(recall))

    f1_score = 2 * precision * recall / (precision + recall)
    print("F1 score {}".format(f1_score))

    return precision, recall, f1_score


def get_prediction_outcome(annotations_dict, predictions, iou_threshold=0.1):
    """
    Determines TPs, FPS and FNs based on the GT annotations, predictions and IoU threshold
    Parameters
    ----------
    annotations : list of AdhesionAnnotation
    predictions : list of list
       Bounding boxes and confidences
    iou_threshold : float

    Returns
    -------
    tps, fps, fns : list

    """
    tps = []
    fps = []
    fns = []

    for slice_id, prediction in predictions.items():
        if slice_id in annotations_dict:
            annotation = annotations_dict[slice_id]
            # Tracks which predictions has been assigned to a TP
            hit_predictions_inds = []
            # Loop over TPs
            for adhesion in annotation.adhesions:
                adhesion.adjust_size()

                max_iou = 0
                max_iou_ind = -1
                # For simplicity for now one predicted bb can correspond to only one TP
                for ind, (bounding_box, confidence) in enumerate(prediction):
                    curr_iou = adhesion.iou(bounding_box)
                    # Do not use predictions that have already been assigned to a TP
                    if curr_iou > max_iou and ind not in hit_predictions_inds:
                        max_iou = curr_iou
                        max_iou_ind = ind

                # If a maximum IoU is greater than the threshold, consider a TP as found
                if max_iou >= iou_threshold:
                    matching_pred = prediction[max_iou_ind]
                    tps.append(matching_pred)
                    hit_predictions_inds.append(max_iou_ind)
                # Otherwise add it as a false negative
                else:
                    fns.append(adhesion)

            # Get false positive
            # Predictions that was not matched with a TP are FPs
            tp_mask = np.full(len(prediction), True, dtype=bool)
            tp_mask[hit_predictions_inds] = False
            fps_left = np.array(prediction)[tp_mask]
            fps += fps_left.tolist()
        else:
            # All predicted bounding boxes are FPs
            fps += prediction

    return tps, fps, fns


def get_confidence_outcome(tps, fps, fns):
    """
    Determines whether prediction with a given confidence is true or false based on the TPs and FPs lists
    Parameters
    ----------
    tps : list
       A list of true positives
    fps : list
       A list of false positives

    Returns
    -------
    outcomes : list
       A list of tuple of confidence and whether its prediction is true
    """
    outcomes = []
    for _, confidence in tps:
        outcomes.append((1, confidence))
        
    for _, confidence in fps:
        outcomes.append((0, confidence))

    for _ in fns:
        outcomes.append((1, 0))
        
    return outcomes



def get_vs_range(visceral_slides):
    # Statistics useful for prediction
    all_vs_values = []
    for visceral_slide in visceral_slides:
        all_vs_values.extend(visceral_slide.values)

    vs_abs_max = np.max(all_vs_values)
    vs_abs_min = np.min(all_vs_values)
    vs_q1 = np.quantile(all_vs_values, 0.25)
    vs_q3 = np.quantile(all_vs_values, 0.75)
    vs_iqr = vs_q3 - vs_q1
    vs_min = vs_q1 - 1.5 * vs_iqr
    vs_max = vs_q3 + 1.5 * vs_iqr

    return vs_min, vs_max


def predict_and_evaluate(visceral_slides, annotations_dict, output_path, bb_size_median=False, vs_expectation=None,
                         expectation_norm_type=VSExpectationNormType.mean_div):
    """
    Performs prediction by visceral slide threshold and evaluates it

    Parameters
    ----------
    annotations
    visceral_slide_path

    Returns
    -------
    metric :
       Evaluation metric
    """
    output_path.mkdir(exist_ok=True, parents=True)

    annotations = annotations_dict.values()

    # Average bounding box size
    bb_mean_size, bb_size_std = bb_size_stat(annotations, is_median=bb_size_median)
    bb_size_max = bb_mean_size[0] + 1.96 * bb_size_std[0], bb_mean_size[1] + 1.96 * bb_size_std[1]

    # Normalize with expectation
    if vs_expectation:
        for vs in visceral_slides:
            vs.norm_with_expectation(vs_expectation[0], vs_expectation[1], expectation_norm_type)

    # vary threshold level
    # Get predictions by visceral slide level threshold
    negative_vs_needed = vs_expectation is not None and expectation_norm_type == VSExpectationNormType.standardize
    vs_min, vs_max = (-np.inf, 0) if negative_vs_needed else get_vs_range(visceral_slides)

    predictions = predict(visceral_slides, bb_mean_size, bb_size_max, (vs_min, vs_max))

    tps, fps, fns = get_prediction_outcome(annotations_dict, predictions, 0.01)

    outcomes = get_confidence_outcome(tps, fps, fns)

    total_patients = len(annotations)

    adhesions_num = 0
    for annotation in annotations:
        adhesions_num += len(annotation.adhesions)

    FP_per_image, FP_per_normal_image, sensitivity, thresholds = y_to_FROC(outcomes, [], total_patients, 1)

    plot_FROC(FP_per_image, sensitivity, output_path)

    recall, precision, thresholds = compute_pr_curves(outcomes)
    ap, precision1, recall1 = compute_ap(recall, precision)

    plot_precision_recall(recall, thresholds, output_path, "Recall")
    plot_precision_recall(precision, thresholds, output_path)

    print("Average precision {}".format(ap))

    tp_conf = [tp[1] for tp in tps]
    mean_tp_conf = np.mean(tp_conf)
    print("Mean TP conf {}".format(mean_tp_conf))

    fp_conf = [fp[1] for fp in fps]
    mean_fp_conf = np.mean(fp_conf)
    print("Mean FP conf {}".format(mean_fp_conf))

    t_test = stats.ttest_ind(tp_conf, fp_conf, equal_var=False)
    print("T-stat {}, p-value {}".format(t_test.statistic, t_test.pvalue))

    # Get binary data
    scores = []
    labels = []
    for slice_full_id, prediction in predictions.items():
        if len(prediction) > 0:
            _, confidence = prediction[0]
            scores.append(confidence)
        else:
            scores.append(0)

        label = 1 if slice_full_id in annotations_dict else 0
        labels.append(label)

    compute_slice_level_ROC(scores, labels, output_path)



def compute_pr_curves(outcomes):
    # Compute precision and recall curves from list of predictions and confidence
    # Sort Predictions
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
            fp = np.sum(y_pred_all_thresholded - y_true_all*y_pred_all_thresholded)

            # Add the corresponding precision and recall values
            recall.append(tp / total_lesions)
            curr_precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
            precision.append(curr_precision)
        else:
            # Extend precision and recall curves
            if (len(recall) > 0):
                recall.append(recall[-1])
                precision.append(precision[-1])

    return recall, precision, thresholds


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    Taken from https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap, mpre, mrec


def compute_slice_level_ROC(thresholds, labels, output_path):

    fpr, tpr, thresholds = roc_curve(labels, thresholds)
    auc_val = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Slide level ROC')
    plt.legend(loc="lower right")
    plt.savefig(output_path / "Slice_ROC.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    return  auc_val


def plot_FROC(FP_per_image, sensitivity, output_path):

    plt.figure()
    plt.plot(FP_per_image, sensitivity)
    plt.xlabel("Mean number of FPs per image")
    plt.ylabel("TPs fraction")
    plt.ylim([0, 1])
    plt.savefig(output_path / "FROC.png", bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_precision_recall(values, confidence, output_path, metric="Precision"):

    plt.figure()
    plt.plot(confidence, values)
    plt.xlabel("Confidence")
    plt.ylabel(metric)
    if metric == "Recall":
        plt.ylim([0, 1])
    plt.savefig(output_path / "{}.png".format(metric), bbox_inches='tight', pad_inches=0)
    plt.show()

# determine bounding box as rect with origin in min x, min y and bottom right angle max x, max y
# for this setup min number of points in the cluster and connectivity threshold
# also IoU - 0.1?

# TODO: document
# Plots normalized visceral slide values against adhesion frequency
def vs_adhesion_likelihood(visceral_slide_path,
                           annotations_path,
                           intervals_num=1000,
                           adhesion_types=[AdhesionType.anteriorWall,
                                           AdhesionType.abdominalCavityContour],
                           plot=False):

    annotations_dict = load_annotations(annotations_path, as_dict=True, adhesion_types=adhesion_types)
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
        plt.axhline(y=0.001, color='r', linestyle='--')
        plt.ylim([0, max_y])
        plt.title("Adhesion")
        plt.xlabel("Normalised visceral slide")
        plt.ylabel("Likelihood")
        plt.savefig("adh_freq_int_{}".format(intervals_num), bbox_inches='tight', pad_inches=0)
        plt.show()

        plt.figure()
        plt.plot(reference_vals, not_adh_likelihood)
        plt.axhline(y=0.001, color='r', linestyle='--')
        plt.ylim([0, max_y])
        plt.title("Not Adhesion")
        plt.xlabel("Normalised visceral slide")
        plt.ylabel("Likelihood")
        plt.savefig("not_adh_freq_int_{}".format(intervals_num), bbox_inches='tight', pad_inches=0)
        plt.show()

    return reference_vals, adh_likelihood, not_adh_likelihood


def vs_value_distr(visceral_slide_path,
                   intervals_num=1000):

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
    plt.savefig("vs_freq_int_{}".format(intervals_num), bbox_inches='tight', pad_inches=0)
    plt.show()


def vs_adhesion_boxplot(visceral_slide_path,
                        annotations_path,
                        adhesion_types=[AdhesionType.anteriorWall,
                                        AdhesionType.abdominalCavityContour]):

    annotations_dict = load_annotations(annotations_path, as_dict=True, adhesion_types=adhesion_types)
    visceral_slides = load_visceral_slides(visceral_slide_path)

    points_in_annotations = []
    points_not_in_annotations = []

    # Calculate frequencies in
    for vs in visceral_slides:
        has_annotation = vs.full_id in annotations_dict
        if not has_annotation:
            points_not_in_annotations.extend(vs.values)
        else:
            annotation = annotations_dict[vs.full_id]
            # For each point check if it intersects any adhesion
            for x, y, vs in zip(vs.x, vs.y, vs.values):
                intersects = False
                for adhesion in annotation.adhesions:
                    intersects = adhesion.contains_point(x, y)
                    if intersects:
                        break

                if intersects:
                    points_in_annotations.append(vs)
                else:
                    points_not_in_annotations.append(vs)

    plt.figure()
    plt.boxplot(points_in_annotations)
    plt.savefig("adh_boxplot_inspexp_norm_avg", bbox_inches='tight', pad_inches=0)
    plt.show()

    print("Adhesions VS stat:")
    print("Median {}".format(np.median(points_in_annotations)))
    print("25% precentile {}".format(np.percentile(points_in_annotations, 25)))
    print("75% precentile {}".format(np.percentile(points_in_annotations, 75)))

    plt.figure()
    plt.boxplot(points_not_in_annotations)
    plt.savefig("no_adh_boxplot_inspexp_norm_avg", bbox_inches='tight', pad_inches=0)
    plt.show()

    print("No Adhesions VS stat:")
    print("Median {}".format(np.median(points_not_in_annotations)))
    print("25% precentile {}".format(np.percentile(points_not_in_annotations, 25)))
    print("75% precentile {}".format(np.percentile(points_not_in_annotations, 75)))

    t_test = stats.ttest_ind(points_in_annotations, points_not_in_annotations, equal_var=False)
    print("T-stat {}, p-value {}".format(t_test.statistic, t_test.pvalue))


def vs_values_boxplot(visceral_slide_path):

    visceral_slides = load_visceral_slides(visceral_slide_path)
    vs_values = []
    for visceral_slide in visceral_slides:
        vs_values.extend(visceral_slide.values)

    plt.figure()
    plt.boxplot(vs_values)
    plt.savefig("vs_boxplot_inspexp_norm_avg", bbox_inches='tight', pad_inches=0)
    plt.show()


def test():
    archive_path = ARCHIVE_PATH
    images_path = archive_path / IMAGES_FOLDER
    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    visceral_slide_path = Path(VISCERAL_SLIDE_PATH)
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    full_segmentation_path = archive_path / FULL_SEGMENTATION_FOLDER / "merged_segmentation"
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    adhesion_types = [AdhesionType.anteriorWall, AdhesionType.abdominalCavityContour]
    annotations = load_annotations(annotations_path, adhesion_types=adhesion_types)
    
    output = Path("threshold_prediction_09_example")
    #cumulative_vs_path = Path("../../data/vs_cum/cumulative_vs_rest_reg_pos")
    cumulative_vs_path = Path("../../data/vs_cum/cumulative_vs_contour_reg_det_full_df")

    """
    int_nums = [100, 200, 300, 400, 500, 600, 653, 700, 800, 900, 1000, 1500, 2000]
    int_nums = [200, 653]
    for int_num in int_nums:
        vs_value_distr(cumulative_vs_path, int_num)
        values, adh_likelihood, not_adh_likelihood = vs_adhesion_likelihood(cumulative_vs_path, annotations_path, int_num, plot=True)
        #print(adh_likelihood.sum())
        #print(not_adh_likelihood.sum())
    """


    int_num = 653
    values, adh_likelihood, not_adh_likelihood = vs_adhesion_likelihood(cumulative_vs_path, annotations_path, int_num)
    likelihood_data = {"values": values, "adh_like": adh_likelihood, "not_adh_like": not_adh_likelihood}

    predict_and_visualize(annotations[:2], cumulative_vs_path, images_path, output, likelihood_data, 0.5, threshold=0.9)
    #prediction_by_threshold(annotations, cumulative_vs_path, output, likelihood_data, 0.5, threshold=0.2)


    #bb_with_threshold(annotations, visceral_slide_path, inspexp_data, images_path)


def test_vs_calc_loading():
    detection_path = Path(DETECTION_PATH)
    images_path = detection_path / IMAGES_FOLDER / TRAIN_FOLDER
    cum_vs_path = detection_path / VS_FOLDER / AVG_NORM_FOLDER / CUMULATIVE_VS_FOLDER
    cum_vs_expectation_path = detection_path / METADATA_FOLDER / CUMULATIVE_VS_EXPECTATION_FILE
    with open(cum_vs_expectation_path, "r+b") as file:
        cum_vs_expectation_dict = pickle.load(file)
        cum_vs_expectation = cum_vs_expectation_dict["means"], cum_vs_expectation_dict["stds"]

    insp_exp_vs_path = detection_path / VS_FOLDER / AVG_NORM_FOLDER / INS_EXP_VS_FOLDER
    insp_exp_vs_expectation_path = detection_path / METADATA_FOLDER / CUMULATIVE_VS_EXPECTATION_FILE
    with open(insp_exp_vs_expectation_path, "r+b") as file:
        insp_exp_vs_expectation_dict = pickle.load(file)
        insp_exp_vs_expectation = insp_exp_vs_expectation_dict["means"], insp_exp_vs_expectation_dict["stds"]

    inspexp_file_path = detection_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    annotations_path = detection_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    adhesion_types = [AdhesionType.anteriorWall, AdhesionType.abdominalCavityContour]
    annotations = load_annotations(annotations_path, adhesion_types=adhesion_types)

    adhesions_stat(annotations)

    annotations_dict = load_annotations(annotations_path, as_dict=True, adhesion_types=adhesion_types)
    visceral_slides = load_visceral_slides(cum_vs_path)
    visceral_slides.sort(key=lambda vs: vs.full_id, reverse=False)

    output = Path(DETECTION_PATH) / "predictions" / "cum_avg_no_avg_expectation" / "cum_vs_warp_contour_norm_avg_rest_region_growing_range_conf_min_vic2_vs2_5_roc_test"

    predict_and_evaluate(visceral_slides, annotations_dict, output, bb_size_median=False)
    #predict_and_visualize(visceral_slides, annotations_dict, images_path, output, bb_size_median=False)

    #vs_values_boxplot(insp_exp_vs_path)
    #vs_adhesion_boxplot(insp_exp_vs_path, annotations_path)

    #int_num = 653
    #vs_value_distr(cum_vs_path, int_num)
    #values, adh_likelihood, not_adh_likelihood = vs_adhesion_likelihood(cum_vs_path, annotations_path, int_num, plot=True)

    # How to integrate folds? Read full ids of slices per fold and filter out
    # extract visceral slides, look at the statistics

    #

    pass


if __name__ == '__main__':
    test_vs_calc_loading()
