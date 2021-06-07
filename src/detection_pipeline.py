import numpy as np
import SimpleITK as sitk
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from adhesions import AdhesionType, Adhesion, load_annotations
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, BB_ANNOTATIONS_EXPANDED_FILE, VISCERAL_SLIDE_FILE
from cinemri.config import ARCHIVE_PATH
from utils import average_bb_size
from contour import get_connected_regions, get_adhesions_prior_coords, get_abdominal_contour_top
from visceral_slide_pipeline import get_inspexp_frames
from froc.deploy_FROC import y_to_FROC
from scipy import stats

VISCERAL_SLIDE_PATH = "../../data/visceral_slide_all/visceral_slide"
PREDICTION_COLOR = (0, 0.8, 0.2)

# thresholding method
# map
# for now only positive classes

# exract annotations
# extract visceral slide for annotations
# threshold (lower 10% ?)
# map

# method to load visceral slide for annotations
# output dict by full id

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


def bb_from_region(region, mean_width, mean_height):
    """
    Calculates bounding box based on the coordinates of the points in the region
    and mean annotations width and height
    
    Parameters
    ----------
    region : list of list
       A list of contour points represented as 2d arrays

    mean_width, mean_height : float
       Mean wide and height of adhesions bounding boxes in annotations

    Returns
    -------
    adhesion : Adhesion
       A bounding box corresponding to the region
    """
    region_centre = region[round(len(region) / 2) - 1]

    x_min = np.min([coord[0] for coord in region])
    x_max = np.max([coord[0] for coord in region])
    width = x_max - x_min + 1
    if width < mean_width:
        origin_x = region_centre[0] - round(mean_width / 2)
        width = mean_width
    else:
        origin_x = x_min

    y_min = np.min([coord[1] for coord in region])
    y_max = np.max([coord[1] for coord in region])
    height = y_max - y_min + 1
    if height < mean_height:
        origin_y = region_centre[1] - round(mean_height / 2)
        height = mean_height
    else:
        origin_y = y_min
    
    return Adhesion([origin_x, origin_y, width, height])


def bb_with_threshold(x, y, slide_normalized, mean_width, mean_height, threshold=0.2, min_region_len=3):
    """
    Predicts adhesions with bounding boxes based on the values of the normalized visceral slide and the
    specified threshold level

    Parameters
    ----------
    x, y : ndarray of int
       x-axis and y-axis components of abdominal cavity contour
    slide_normalized : ndarray of float
       Normalized values of visceral slide corresponding to the coordinates of abdominal cavity contour
    mean_width, mean_height : float
       Mean wide and height of adhesions bounding boxes in annotations
    threshold : float
       A percentile of a low slide
    min_region_len : int
       Minumum length of the connected region if low slide to be considered

    Returns
    -------
    bounding_boxes : list of Adhesion
       A list of bounding boxes predicted based on visceral slide values
    """

    # Extract top coordinates

    """
    x_top, y_top = get_abdominal_contour_top(x, y)

    coords = np.column_stack((x, y)).tolist()
    top_coords = np.column_stack((x_top, y_top)).tolist()
    top_excluded_inds = [ind for ind, coord in enumerate(coords) if coord not in top_coords]

    # Remove top coordinates
    x = x[top_excluded_inds]
    y = y[top_excluded_inds]
    slide_normalized = slide_normalized[top_excluded_inds]
    """

    x_prior, y_prior = get_adhesions_prior_coords(x, y)

    coords = np.column_stack((x, y)).tolist()
    prior_coords = np.column_stack((x_prior, y_prior)).tolist()
    prior_inds = [ind for ind, coord in enumerate(coords) if coord in prior_coords]

    x = x[prior_inds]
    y = y[prior_inds]
    slide_normalized = slide_normalized[prior_inds]

    # Find lowest threshold % in normalized slide
    slide_normalized_sorted = np.sort(slide_normalized)
    threshold_ind = int(len(slide_normalized) * threshold)

    low_slide_threshold = slide_normalized_sorted[threshold_ind]

    lowest_slide = slide_normalized_sorted.min()
    norm_const = low_slide_threshold / (low_slide_threshold - lowest_slide)

    # Coordinates and values of low slide
    low_slide_regions = slide_normalized <= low_slide_threshold
    low_slide_x = x[low_slide_regions]
    low_slide_y = y[low_slide_regions]
    low_slide_values = slide_normalized[low_slide_regions]

    low_slide_coords = np.column_stack((low_slide_x, low_slide_y, low_slide_values))
    low_slide_regions = get_connected_regions(low_slide_coords)
    # Add threshold to regions

    # Filter regions by min length
    # Need a set of points with len
    low_slide_regions = [region for region in low_slide_regions if len(region) >= min_region_len]

    # Get central point of each region to generate a bounding box
    bounding_boxes = [bb_from_region(region, mean_width, mean_height) for region in low_slide_regions]

    vs_min = np.min(slide_normalized)
    vs_max = np.max(slide_normalized)

    """
    adhesion_likelihoods = [adhesion_likelihood(region, vs_min, vs_max, low_slide_threshold) for region in low_slide_regions]
    confidence = adhesion_likelihoods.copy()
    #confidence = [np.power(like, 1. / len(region)) for like, region in zip(adhesion_likelihoods, low_slide_regions)]
    """

    # Compute confidence for each prediction based on the region average index
    mean_slides = []
    for region in low_slide_regions:
        region_slides = [comp[2] for comp in region]
        mean_region_slide = np.mean(region_slides)
        mean_slides.append(mean_region_slide)
    confidence = [norm_const * (1 - mean_slide / low_slide_threshold) for mean_slide in mean_slides]

    prediction = [(bb, conf) for bb, conf in zip(bounding_boxes, confidence)]
    return prediction


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


def visualize_gt_vs_prediction(annotation, prediction, x, y, slide_normalized, insp_frame, file_path=None):
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
    # Plot regions of slow slide
    # plt.scatter(low_slide_x, low_slide_y, s=10, color="k", marker="x")
    ax = plt.gca()
    for adhesion in annotation.adhesions:
        adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                  linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(adhesion_rect)

    for adhesion, confidence in prediction:
        adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                  linewidth=1.5, edgecolor=PREDICTION_COLOR, facecolor='none')
        ax.add_patch(adhesion_rect)
        plt.text(adhesion.origin_x, adhesion.origin_y - 3, "{:.3f}".format(confidence), c=PREDICTION_COLOR, fontweight='semibold')
    plt.axis("off")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()


def predict_and_visualize(annotations, visceral_slide_path, images_path, output_path, inspexp_data=None,
                          threshold=0.2, min_region_len=5):

    output_path.mkdir(exist_ok=True)

    visceral_slide_dict = extract_visceral_slide_dict(annotations, visceral_slide_path)
    # Average bounding box size
    mean_width, mean_height = average_bb_size(annotations)

    for annotation in annotations:
        visceral_slide_data = visceral_slide_dict[annotation.full_id]
        x, y = visceral_slide_data["x"], visceral_slide_data["y"]
        visceral_slide = visceral_slide_data["slide"]
        slide_normalized = np.abs(visceral_slide) / np.abs(visceral_slide).max()

        prediction = bb_with_threshold(x, y, slide_normalized, mean_width, mean_height, threshold, min_region_len)

        if inspexp_data is None:
            # Assume it is cumulative VS and take the one before the last one frame
            slice_path = annotation.build_path(images_path)
            slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
            frame = slice[-2]
        else:
            # Extract inspiration frame
            try:
                frame, _ = get_inspexp_frames(annotation.slice, inspexp_data, images_path)
            except:
                print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(slice.patient_id,
                                                                                           slice.scan_id,
                                                                                           slice.slice_id))

        file_path = output_path / (annotation.full_id + ".png")
        visualize_gt_vs_prediction(annotation, prediction, x, y, slide_normalized, frame, file_path)


# TODO: should we also output confidence here?
def predict(full_ids, visceral_slide_dict, mean_width, mean_height, threshold=0.2, min_region_len=3):
    """
    Predicts adhesions based on visceral slide values for the provided full ids of slices

    Parameters
    ----------
    full_ids : list of str
       Full ids of slices in the format patientID_scanID_slice_ID
    visceral_slide_dict : dict
       A dictionary that contains the coordinates of abdominal cavity contour and
       the corresponding visceral slide

    Returns
    -------
    prediction : dict
       A dictionary with full ids of scans used as keys and predicted bounding boxes as values
    """

    predictions = {}

    for full_id in full_ids:
        visceral_slide_data = visceral_slide_dict[full_id]
        x, y = visceral_slide_data["x"], visceral_slide_data["y"]
        visceral_slide = visceral_slide_data["slide"]
        slide_normalized = np.abs(visceral_slide) / np.abs(visceral_slide).max()

        prediction = bb_with_threshold(x, y, slide_normalized, mean_width, mean_height, threshold, min_region_len)
        predictions[full_id] = prediction

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
                scan_data = patient_data[annotation.scan_id]
                inspexp_frames = scan_data[annotation.slice_id]
            except:
                print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.scan_id,
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


def get_prediction_outcome(annotations, predictions, iou_threshold=0.1):
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
    for annotation in annotations:
        prediction = predictions[annotation.full_id]
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


def prediction_by_threshold(annotations, visceral_slide_path, output_path, threshold=0.2):
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

    visceral_slide_dict = extract_visceral_slide_dict(annotations, visceral_slide_path)

    # Average bounding box size
    mean_width, mean_height = average_bb_size(annotations)
    full_ids = [annotation.full_id for annotation in annotations]

    # vary threshold level
    # Get predictions by visceral slide level threshold
    predictions = predict(full_ids, visceral_slide_dict, mean_width, mean_height, threshold, 5)

    archive_path = Path(ARCHIVE_PATH)
    images_path = archive_path / IMAGES_FOLDER
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    tps, fps, fns = get_prediction_outcome(annotations, predictions, 0.01)
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
            y_pred_all_thresholded                  = np.zeros_like(y_pred_all)
            y_pred_all_thresholded[y_pred_all > th] = 1
            tp     = np.sum(y_true_all * y_pred_all_thresholded)
            fp     = np.sum(y_pred_all_thresholded - y_true_all*y_pred_all_thresholded)

            # Add the corresponding precision and recall values
            curr_precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
            recall.append(tp / total_lesions)
            precision.append(curr_precision)
        else:
            # Extend precision and recall curves
            if (len(recall)>0):
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


def plot_FROC(FP_per_image, sensitivity, output_path):

    plt.figure()
    plt.plot(FP_per_image, sensitivity)
    plt.xlabel("Mean number of FPs per image")
    plt.ylabel("TPs fraction")
    plt.savefig(output_path / "FROC.png", bbox_inches='tight', pad_inches=0)
    #plt.show()


def plot_precision_recall(values, confidence, output_path, metric="Precision"):

    plt.figure()
    plt.plot(confidence, values)
    plt.xlabel("Confidence")
    plt.ylabel(metric)
    plt.savefig(output_path / "{}.png".format(metric), bbox_inches='tight', pad_inches=0)
    plt.show()

# determine bounding box as rect with origin in min x, min y and bottom right angle max x, max y
# for this setup min number of points in the cluster and connectivity threshold
# also IoU - 0.1?

# TODO: document
# Plots normalized visceral slide values against adhesion frequency
def vs_intensity_stat(visceral_slide_path, annotations_path, intervals_num=1000):
    adhesion_types = [AdhesionType.anteriorWall, AdhesionType.abdominalCavityContour]
    annotations = load_annotations(annotations_path, adhesion_types=adhesion_types)

    visceral_slide_dict = extract_visceral_slide_dict(annotations, visceral_slide_path)
    # Binning
    intervals = np.linspace(0, 1, intervals_num + 1)
    reference_vals = []
    freq_dict = {}
    for i in range(intervals_num):
        interval_start = intervals[i]
        interval_end = intervals[i + 1]
        interval_middle = (interval_start + interval_end) / 2
        reference_vals.append(interval_middle)
        freq_dict[i] = 0

    reference_vals = np.array(reference_vals)

    for annotation in annotations:
        visceral_slide_data = visceral_slide_dict[annotation.full_id]
        contour_x, contour_y = visceral_slide_data["x"], visceral_slide_data["y"]
        visceral_slide = visceral_slide_data["slide"]
        slide_normalized = visceral_slide / np.max(visceral_slide)

        for adhesion in annotation.adhesions:
            # For each point check if it intersects an adhesion
            for x, y, vs in zip(contour_x, contour_y, slide_normalized):
                intersects = adhesion.contains_point(x, y)
                if intersects:
                    # Find the reference value closest to VS and increase the counter
                    diff = reference_vals - vs
                    index = np.argmin(np.abs(diff))
                    freq_dict[index] = freq_dict[index] + 1

    # Now extract ordered frequencies
    frequencies = []
    for i in range(intervals_num):
        frequencies.append(freq_dict[i])

    plt.figure()
    plt.plot(reference_vals, frequencies)
    plt.xlabel("Normalised visceral slide")
    plt.ylabel("Adhesion frequency")
    plt.savefig("adh_freq_int_{}".format(intervals_num), bbox_inches='tight', pad_inches=0)
    plt.show()


def test():
    archive_path = Path(ARCHIVE_PATH)
    images_path = archive_path / IMAGES_FOLDER
    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    visceral_slide_path = Path(VISCERAL_SLIDE_PATH)
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    full_segmentation_path = archive_path / "full_segmentation" / "merged_segmentation"
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    adhesion_types = [AdhesionType.anteriorWall, AdhesionType.abdominalCavityContour]
    annotations = load_annotations(annotations_path, adhesion_types=adhesion_types)
    
    output = Path("threshold_prediction_cumulative_contour_reg_th20_mean")
    #cumulative_vs_path = Path("../../data/cumulative_vs_rest_reg")
    cumulative_vs_path = Path("../../data/cumulative_vs_contour_reg")

    int_nums = [100, 200, 300, 400, 500, 600, 653, 700, 800, 900, 1000, 1500, 2000]
    for int_num in int_nums:
        vs_intensity_stat(cumulative_vs_path, annotations_path, int_num, prior=True)
    
    # annotations_stat(annotations)
    # predict_and_visualize(annotations, cumulative_vs_path, images_path, output, threshold=0.2)
    # prediction_by_threshold(annotations, cumulative_vs_path, output, threshold=0.2)

    #bb_with_threshold(annotations, visceral_slide_path, inspexp_data, images_path)


if __name__ == '__main__':
    test()
