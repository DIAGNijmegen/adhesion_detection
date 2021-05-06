import numpy as np
import SimpleITK as sitk
import pickle
import json
from matplotlib.patches import Rectangle
from adhesions import AdhesionType, Adhesion, load_annotations, get_abdominal_contour_top
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, BB_ANNOTATIONS_EXPANDED_FILE
from pathlib import Path
from cinemri.config import ARCHIVE_PATH
import matplotlib.pyplot as plt

VISCERAL_SLIDE_PATH = "../../data/visceral_slide_all/visceral_slide"

# thresholding method
# map
# for now only positive classes

# exract annotations
# extract visceral slide for annotations
# threshold (lower 10% ?)
# map

# method to load visceral slide for annotations
# output dict by full id

def load_visceral_slide(annotations, visceral_slide_path):
    visceral_slide_dict = {}
    for annotation in annotations:
        visceral_slide_results_path = annotation.build_path(visceral_slide_path, extension="")
        if visceral_slide_results_path.exists():
            # Load the computed visceral slide
            visceral_slide_file_path = visceral_slide_results_path / "visceral_slide.pkl"
            with open(str(visceral_slide_file_path), "r+b") as file:
                visceral_slide_data = pickle.load(file)
                visceral_slide_dict[annotation.full_id] = visceral_slide_data

    return visceral_slide_dict


def annotations_stat(annotations):
    """
    Reports statistics of bounding box annotations

    Parameters
    ----------
    annotations : list of AdhesionAnnotation
        List of annotations for which statistics should be computed
    """

    widths = []
    heights = []
    largest_bb = annotations[0].adhesions[0]
    smallest_bb = annotations[0].adhesions[0]

    for annotation in annotations:
        for adhesion in annotation.adhesions:
            widths.append(adhesion.width)
            heights.append(adhesion.height)

            curr_perim = adhesion.width + adhesion.height

            if curr_perim > largest_bb.width + largest_bb.height:
                largest_bb = adhesion

            if curr_perim < smallest_bb.width + smallest_bb.height:
                smallest_bb = adhesion

    print("Minimum width: {}".format(np.min(widths)))
    print("Minimum height: {}".format(np.min(heights)))

    print("Maximum width: {}".format(np.max(widths)))
    print("Maximum height: {}".format(np.max(heights)))

    print("Mean width: {}".format(np.mean(widths)))
    print("Mean height: {}".format(np.mean(heights)))

    print("Median width: {}".format(np.median(widths)))
    print("Median height: {}".format(np.median(heights)))

    print("Smallest annotation, x_o: {}, y_o: {}, width: {}, height: {}".format(smallest_bb.origin_x, smallest_bb.origin_y, smallest_bb.width, smallest_bb.height))
    print("Largest annotation, width: {}, height: {}".format(largest_bb.width, largest_bb.height))


# TODO: training set
def average_bb_size(annotations):
    """
    Computes an average adhesion boundig box size

    Parameters
    ----------
    annotations : list of AdhesionAnnotation
       List of annotations for which an average size of a bounding box should be determined

    Returns
    -------
    width, height : int
       A tuple of average width and height in adhesion annotations with bounding boxes
    """

    widths = []
    heights = []

    for annotation in annotations:
        for adhesion in annotation.adhesions:
            widths.append(adhesion.width)
            heights.append(adhesion.height)

    return np.mean(widths), np.mean(heights)


# need to get all coordinates in regions
def get_connected_regions(contour_subset_coords, connectivity_threshold=5):
    """
    Given a subset of contour coordinates returns a list of connected regions
    considering the specified connectivity threshold

    Parameters
    ----------
    contour_subset_coords : list of list
       A list containing a subset of coordinates of a contour
    connectivity_threshold : int, default=5
       Threshold which indicates the maximum difference in x component allowed between
       the subsequent coordinates of a contour to be considered connected

    Returns
    -------
    regions : list of list
       A list of lists of coordinates that belong to connected regions. Length of regions might vary

    """

    regions = []
    coord_prev = contour_subset_coords[0]
    coords_num = contour_subset_coords.shape[0]
    current_region = [coord_prev]
    for index in range(1, coords_num):
        coord_curr = contour_subset_coords[index]
        distance = np.sqrt((coord_curr[0] - coord_prev[0])**2 + (coord_curr[1] - coord_prev[1])**2)
        if distance > connectivity_threshold:
            regions.append(current_region)
            current_region = []
        coord_prev = coord_curr
        current_region.append(coord_prev)

    regions.append(current_region)

    return regions


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
       An annotation corresponding to the region

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
    x_top, y_top = get_abdominal_contour_top(x, y)

    coords = np.column_stack((x, y)).tolist()
    top_coords = np.column_stack((x_top, y_top)).tolist()
    top_excluded_inds = [ind for ind, coord in enumerate(coords) if coord not in top_coords]

    # Remove top coordinates
    x = x[top_excluded_inds]
    y = y[top_excluded_inds]
    slide_normalized = slide_normalized[top_excluded_inds]

    # Find lowest threshold % in normalized slide
    slide_normalized_sorted = np.sort(slide_normalized)
    threshold_ind = int(len(slide_normalized) * threshold)

    low_slide_threshold = slide_normalized_sorted[threshold_ind]

    # Coordinates and values of low slide
    low_slide_regions = slide_normalized <= low_slide_threshold
    low_slide_x = x[low_slide_regions]
    low_slide_y = y[low_slide_regions]

    low_slide_coords = np.column_stack((low_slide_x, low_slide_y))
    low_slide_regions = get_connected_regions(low_slide_coords)

    # Filter regions by min length
    low_slide_regions = [region for region in low_slide_regions if len(region) >= min_region_len]

    # Get central point of each region to generate a bounding box
    bounding_boxes = [bb_from_region(region, mean_width, mean_height) for region in low_slide_regions]

    return bounding_boxes


def visualize_gt_vs_prediction(annotation, bounding_boxes, x, y, slide_normalized, insp_frame, file_path=None):
    """
    Visualises ground truth annotations vs prediction together with visceral slide
    on the inspiration frame

    Parameters
    ----------
    annotation : AdhesionAnnotation
        A ground truth annotation for a slice
    bounding_boxes : list of Adhesion
       A list of predicted bounding boxes
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

    for adhesion in bounding_boxes:
        adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                  linewidth=1.5, edgecolor='g', facecolor='none')
        ax.add_patch(adhesion_rect)
    plt.axis("off")

    if file_path is not None:
        plt.savefig(file_path)
    else:
        plt.show()
    
    plt.close()


def predict_and_visualize(annotations, visceral_slide_path, images_path, inspexp_data, output_path,
                          threshold=0.2, min_region_len=3):

    output_path.mkdir(exist_ok=True)

    visceral_slide_dict = load_visceral_slide(annotations, visceral_slide_path)
    # Average bounding box size
    mean_width, mean_height = average_bb_size(annotations)

    for annotation in annotations:
        visceral_slide_data = visceral_slide_dict[annotation.full_id]
        x, y = visceral_slide_data["x"], visceral_slide_data["y"]
        visceral_slide = visceral_slide_data["slide"]
        slide_normalized = np.abs(visceral_slide) / np.abs(visceral_slide).max()

        bounding_boxes = bb_with_threshold(x, y, slide_normalized, mean_width, mean_height, threshold, min_region_len)

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

        file_path = output_path / (annotation.full_id + ".png")

        visualize_gt_vs_prediction(annotation, bounding_boxes, x, y, slide_normalized, insp_frame, file_path)


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

        bounding_boxes = bb_with_threshold(x, y, slide_normalized, mean_width, mean_height, threshold, min_region_len)
        predictions[full_id] = bounding_boxes

    return predictions

# Some annotations are very small - setup min annotation size
# Min width and height - 15 +

# TODO: understand adequate evaluation metric
# Recall and precision since we do not have confidence score?
# F1 score
# Maybe confidence - inverse of length? Ask supervisors about it
def evaluate(annotations, predictions, iou_threshold=0.1, visceral_slide_dict=None, images_path=None, inspexp_data=None):

    found_tp = []
    tp_num = 0
    fps = []
    fns = []
    prediction_num = 0
    for annotation in annotations:
        prediction = predictions[annotation.full_id]
        hit_predictions_inds = []
        tp_num += len(annotation.adhesions)
        prediction_num += len(prediction)
        for adhesion in annotation.adhesions:
            adhesion.adjust_size()

            max_iou = 0
            max_iou_ind = -1
            # For simplicity for now one predicted bb can correspond to only one TP
            for ind, bounding_box in enumerate(prediction):
                curr_iou = adhesion.iou(bounding_box)
                if curr_iou > max_iou and ind not in hit_predictions_inds:
                    max_iou = curr_iou
                    max_iou_ind = ind

            if max_iou >= iou_threshold:
                found_tp.append((adhesion, bounding_box))
                hit_predictions_inds.append(max_iou_ind)
            else:
                fns.append(adhesion)

        # get false positive
        tp_mask = np.full(len(prediction), True, dtype=bool)
        tp_mask[hit_predictions_inds] = False
        fps_left = np.array(prediction)[tp_mask]
        fps.append(fps_left)

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
    precision = len(found_tp) / prediction_num
    print("Precision {}".format(precision))

    # How many of TP are found
    recall = len(found_tp) / tp_num
    print("Recall {}".format(recall))

    f1_score = 2 * precision * recall / (precision + recall)
    print("F1 score {}".format(f1_score))

    return precision, recall, f1_score


def prediction_by_threshold(annotations, visceral_slide_path):
    """
    Performs prediction by visceral slide threshold and evaluates it

    Parameters
    ----------
    annotations
    visceral_slide_path

    Returns
    -------
    metric : float
       Evaluation metric

    """

    visceral_slide_dict = load_visceral_slide(annotations, visceral_slide_path)

    # Average bounding box size
    mean_width, mean_height = average_bb_size(annotations)
    full_ids = [annotation.full_id for annotation in annotations]

    # vary threshold level
    # Get predictions by visceral slide level threshold
    predictions = predict(full_ids, visceral_slide_dict, mean_width, mean_height, 0.1, 5)

    archive_path = Path(ARCHIVE_PATH)
    images_path = archive_path / IMAGES_FOLDER
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    #evaluate(annotations, predictions, 0.01, visceral_slide_dict, images_path, inspexp_data)
    evaluate(annotations, predictions, 0.01)


# determine bounding box as rect with origin in min x, min y and bottom right angle max x, max y
# for this setup min number of points in the cluster and connectivity threshold
# also IoU - 0.1?


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

    adhesion_types = [AdhesionType.anteriorWall.value, AdhesionType.abdominalCavityContour.value]
    annotations = load_annotations(annotations_path, adhesion_types=adhesion_types)
    
    output = Path("threshold_prediction")
    
    #annotations_stat(annotations)
    #predict_and_visualize(annotations, visceral_slide_path, images_path, inspexp_data, output)
    prediction_by_threshold(annotations, visceral_slide_path)

    #bb_with_threshold(annotations, visceral_slide_path, inspexp_data, images_path)


if __name__ == '__main__':
    test()
