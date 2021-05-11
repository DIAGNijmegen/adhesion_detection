# Tasks
# 1. Extract the subset of scans with annotations and visceral slide information
# 2. Visualize displacement field as RGB with B = 0, see if it brings any insight
# 3. Obtain data for detection as registration between expiration and inspiration frames and visceral slide detection.
# Visualize example. Add negative examples
# 4. Train/ test split, training with YOLOv5s
# 5. See if there are any insights from YOLOv5 training results
# 6. Build custom pipeline

import random
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from visceral_slide import VisceralSlideDetector
from adhesions import load_annotated_slices, load_annotations, load_negative_patients, AdhesionType
from cinemri.config import ARCHIVE_PATH
from cinemri.contour import get_contour
from cinemri.utils import get_image_orientation
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, SEPARATOR, BB_ANNOTATIONS_EXPANDED_FILE
from skimage import io
from sklearn.model_selection import KFold

# detection_input_whole folder contains adhesions of all types
# detection_input_contour folder contains adhesions along the abdominal cavity contour

# add possibility to extract scans adjacent to front abdominal wall or abdominal cavity contour (enum)
# save as metadata?

def get_insp_exp_indices(slice, inspexp_data):

    # Extract inspiration and expiration frames for the slice
    try:
        patient_data = inspexp_data[slice.patient_id]
        scan_data = patient_data[slice.scan_id]
        inspexp_frames = scan_data[slice.slice_id]
        insp_frame_index = inspexp_frames[0]
        exp_frame_index = inspexp_frames[1]
        return insp_frame_index, exp_frame_index
    except:
        print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(slice.patient_id,
                                                                                   slice.scan_id,
                                                                                   slice.slice_id))
        return None, None


def get_insp_exp_frames_and_masks(slice, insp_index, exp_index, images_path, masks_path):
    # load image
    slice_path = slice.build_path(images_path)
    slice_array = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
    insp_frame = slice_array[insp_index].astype(np.uint32)
    exp_frame = slice_array[exp_index].astype(np.uint32)

    # load mask
    mask_path = slice.build_path(masks_path)
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    insp_mask = mask_array[insp_index]
    exp_mask = mask_array[exp_index]

    return insp_frame, insp_mask, exp_frame, exp_mask


def get_pseudo_image(deformation_field, x, y, visceral_slide):

    height, width = deformation_field.shape[0], deformation_field.shape[1]

    visceral_slide_channel = np.zeros((height, width))
    for x_coord, y_coord, slide_value in zip(x, y, visceral_slide):
        visceral_slide_channel[y_coord, x_coord] = slide_value

    pseudo_image = np.zeros((height, width, 3))
    pseudo_image[:, :, 0] = deformation_field[:, :, 0]
    pseudo_image[:, :, 1] = -deformation_field[:, :, 1]
    pseudo_image[:, :, 2] = visceral_slide_channel
    return pseudo_image


def save_visualization(annotations_path,
                       inspexp_path,
                       images_path,
                       masks_path,
                       output_path=None):

    # load annotated slices
    annotated_slices = load_annotated_slices(annotations_path)

    for ind, slice in enumerate(annotated_slices):
        visualize_deformation_and_annotations(slice, inspexp_path, images_path, masks_path, output_path)


def visualize_deformation_and_annotations(slice,
                                          inspexp_path,
                                          images_path,
                                          masks_path,
                                          output_path=None):

    # load inspiration and expiration data
    with open(inspexp_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    print("Annotation for patient {}, slice {}".format(slice.patient_id, slice.slice_id))

    if output_path is not None:
        target_folder = output_path / SEPARATOR.join([slice.patient_id, slice.scan_id, slice.slice_id])
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        target_folder = None

    # Extract inspiration and expiration frames for the slice
    insp_index, exp_index = get_insp_exp_indices(slice, inspexp_data)

    if insp_index is None:
        return

    insp_frame, insp_mask, exp_frame, exp_mask = get_insp_exp_frames_and_masks(slice,
                                                                               insp_index,
                                                                               exp_index,
                                                                               images_path,
                                                                               masks_path)

    # Save or show masked inspiration and expiration frames
    plt.figure()
    plt.imshow(insp_frame, cmap="gray")
    masked = np.ma.masked_where(insp_mask == 0, insp_mask)
    plt.imshow(masked, cmap='autumn', alpha=0.2)
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "insp_frame.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    plt.figure()
    plt.imshow(exp_frame, cmap="gray")
    masked = np.ma.masked_where(exp_mask == 0, exp_mask)
    plt.imshow(masked, cmap='autumn', alpha=0.2)
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "exp_frame.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


    # get visceral slide
    x, y, visceral_slide = VisceralSlideDetector().get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask)
    # get deformation field
    deformation_field = VisceralSlideDetector().get_full_deformation_field(exp_frame, insp_frame, exp_mask,
                                                                           insp_mask)


    pseudo_image = get_pseudo_image(deformation_field, x, y, visceral_slide)

    # Visualize deformation as vector field
    deformation_field = np.flipud(deformation_field)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(deformation_field[:, :, 0], -deformation_field[:, :, 1])
    ax.set_aspect(1)
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "deformation_field_vector.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    # Transform to correctly plot as an image
    # Remove negative values and normalize deformation field
    pseudo_image_display = pseudo_image
    pseudo_image_display[:, :, :2] = pseudo_image_display[..., :2] - pseudo_image_display[..., :2].min()
    pseudo_image_display[:, :, :2] /= pseudo_image_display[..., :2].max()
    # Remove take abs value of visceral slide and normalize
    pseudo_image_display[:, :, 2] = np.abs(pseudo_image_display[..., 2])
    pseudo_image_display[:, :, 2] /= pseudo_image_display[..., 2].max()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pseudo_image_display)
    ax.set_aspect(1)
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "deformation_field_image.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    # Visualize deformation as 3 channels image and add tangent vectors of abdominal cavity contour
    x, y, u, v = get_contour(insp_mask)
    # take each 10th vector and invert v to properly display it
    x_vis, y_vis, u_vis, v_vis = x[::10], y[::10], u[::10], -np.array(v[::10])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pseudo_image_display)
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "deformation_field_image.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    # Add abdominal cavity contour tangent vectors
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pseudo_image_display)
    ax.quiver(x_vis, y_vis, u_vis, v_vis, color=(0, 0, 0))
    ax.set_aspect(1)
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "deformation_field_image_vect.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    # Visualize each component of deformation field separately
    plt.figure()
    plt.imshow(pseudo_image_display[:, :, 1])
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "deformation_field_vertical.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    plt.figure()
    plt.imshow(pseudo_image_display[:, :, 0])
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "deformation_field_horizontal.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    # Visualize visceral slide
    plt.figure()
    plt.imshow(pseudo_image_display[:, :, 2])
    plt.axis('off')
    if target_folder is not None:
        plt.savefig(target_folder / "visceral_slide.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


# For now just visualize displacement field (and save data for all annotations for future)
def extract_annotated_slices(annotations_path, inspexp_path, images_path, masks_path, output_path):

    positive_path = output_path / "positive"
    positive_path.mkdir(exist_ok=True, parents=True)

    annotated_slices = load_annotated_slices(annotations_path)
    print("Number of annotated slices {}".format(len(annotated_slices)))

    # load inspiration and expiration data
    with open(inspexp_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    for slice in annotated_slices:
        print("{} {} {}".format(slice.patient_id, slice.scan_id, slice.slice_id))

        insp_index, exp_index = get_insp_exp_indices(slice, inspexp_data)

        if insp_index is None:
            continue

        insp_frame, insp_mask, exp_frame, exp_mask = get_insp_exp_frames_and_masks(slice,
                                                                                   insp_index,
                                                                                   exp_index,
                                                                                   images_path,
                                                                                   masks_path)

        height, width = insp_frame.shape[0], insp_frame.shape[1]

        # TODO: maybe return VS and DF together
        # get visceral slide
        x, y, visceral_slide = VisceralSlideDetector().get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask)
        # get deformation field
        deformation_field = VisceralSlideDetector().get_full_deformation_field(exp_frame, insp_frame, exp_mask,
                                                                               insp_mask)

        pseudo_image = get_pseudo_image(deformation_field, x, y, visceral_slide)

        # Save without normalization
        input_name = SEPARATOR.join([slice.patient_id, slice.scan_id, slice.slice_id])
        input_path = positive_path / (input_name + ".npy")
        np.save(input_path, pseudo_image)


# For now just visualize displacement field (and save data for all annotations for future)
def extract_negative_samples(archive_path, output_path):
    # now we have only 6 negative patients
    # and sample all ASL slices

    negative_path = output_path / "negative"
    negative_path.mkdir(exist_ok=True, parents=True)

    images_path = archive_path / IMAGES_FOLDER
    masks_path = archive_path / "full_segmentation" / "merged_segmentation"
    inspexp_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME

    # load inspiration and expiration data
    with open(inspexp_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    negative_patients = load_negative_patients(archive_path)
    # Extract slices, check if the orientation is correct
    for patient in negative_patients:
        for slice in patient.cinemri_slices:
            print("{} {} {}".format(slice.patient_id, slice.scan_id, slice.slice_id))

            # Check depth and orientation
            slice_path = slice.build_path(images_path)
            slice_image = sitk.ReadImage(str(slice_path))

            # Check that a slice is valid
            depth = slice_image.GetDepth()
            if depth == 30 and get_image_orientation(slice_image) == "ASL":

                insp_index, exp_index = get_insp_exp_indices(slice, inspexp_data)

                if insp_index is None:
                    continue

                insp_frame, insp_mask, exp_frame, exp_mask = get_insp_exp_frames_and_masks(slice,
                                                                                           insp_index,
                                                                                           exp_index,
                                                                                           images_path,
                                                                                           masks_path)

                # Check anatomical orientation
                # Extract DF and visceral slide data and save to disk
                # get visceral slide
                x, y, visceral_slide = VisceralSlideDetector().get_visceral_slide(insp_frame, insp_mask, exp_frame,
                                                                                  exp_mask)
                # get deformation field
                deformation_field = VisceralSlideDetector().get_full_deformation_field(exp_frame, insp_frame, exp_mask,
                                                                                       insp_mask)

                pseudo_image = get_pseudo_image(deformation_field, x, y, visceral_slide)

                # Save without normalization
                input_name = SEPARATOR.join([slice.patient_id, slice.scan_id, slice.slice_id])
                input_path = negative_path / (input_name + ".npy")
                np.save(input_path, pseudo_image)


def annotations_to_yolov5(annotations_path,
                          labels_path,
                          adhesion_types=[AdhesionType.anteriorWall.value,
                                          AdhesionType.abdominalCavityContour.value,
                                          AdhesionType.inside.value]):

    labels_path.mkdir(exist_ok=True, parents=True)
    annotations = load_annotations(annotations_path, adhesion_types=adhesion_types)

    for annotation in annotations:
        print("Converting annotation {} {} {}".format(annotation.patient_id, annotation.scan_id, annotation.slice_id))

        # Get height and width of scans
        slice_path = annotation.build_path(images_path)
        frame = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))[0]
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        slice_id = SEPARATOR.join([annotation.patient_id, annotation.scan_id, annotation.slice_id])
        labels_file_path = labels_path / (slice_id + ".txt")

        annotation_strings = []
        for adhesion in annotation.adhesions:
            x_center, y_center = adhesion.center
            width, height = adhesion.width, adhesion.height

            # Normalize center and shape by frame shape
            x_center_norm = x_center / frame_width
            y_center_norm = y_center / frame_height
            width = width / frame_width
            height = height / frame_height

            bb_str = " ".join(["0", str(x_center_norm), str(y_center_norm), str(width), str(height)])
            annotation_strings.append(bb_str)

        with open(labels_file_path, "x") as labels_file:
            labels_file.write("\n".join(annotation_strings))


def extract_detection_input_subset(detection_input_whole_path,
                                   detection_input_subset_path,
                                   annotations_path,
                                   adhesion_types=[AdhesionType.anteriorWall.value,
                                                   AdhesionType.abdominalCavityContour.value,
                                                   AdhesionType.inside.value]):

    whole_positive_path = detection_input_whole_path / "positive"
    subset_positive_path = detection_input_subset_path / "positive"
    subset_positive_path.mkdir(exist_ok=True, parents=True)

    slices_subset = load_annotated_slices(annotations_path, adhesion_types)
    slices_ids = [slice.full_id for slice in slices_subset]

    whole_files_glob = whole_positive_path.glob("*.npy")
    for whole_file in whole_files_glob:
        file_id = whole_file.stem
        if file_id in slices_ids:
            dest_path = subset_positive_path / whole_file.name
            shutil.copy(whole_file, dest_path)


# Conversion of .npy files containing negative values to RGB
def npy_to_jpg(npy_path,
               jpg_path,
               whole_data_range=True):

    jpg_path.mkdir(exist_ok=True)

    npy_files_glob = npy_path.glob("*.npy")
    npy_file_paths = [file_path for file_path in npy_files_glob]

    if whole_data_range:
        # First get min and max across the whole dataset (to highlight scans with low motion)
        global_min = np.inf
        global_max = -np.inf

        for npy_file_path in npy_file_paths:
            npy_file = np.load(npy_file_path)
            current_min, current_max = npy_file.min(), npy_file.max()
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

        print("Global minimum {}".format(global_min))
        print("Global maximum {}".format(global_max))
    else:
        global_min = global_max = None


    for npy_file_path in npy_file_paths:
        npy_file = np.load(npy_file_path)

        # convert to RGB
        rgb_image = npy_file.copy()
        norm_min = global_min if global_min is not None else rgb_image.min()
        norm_max = global_max if global_max is not None else rgb_image.max()
        # Map to [0, 255] range
        rgb_image = 255 * (rgb_image - norm_min) / (norm_max - norm_min)
        # Convert to uint8
        rgb_image = rgb_image.astype(np.uint8)

        jpg_file_path = jpg_path / (npy_file_path.stem + ".jpg")
        io.imsave(str(jpg_file_path), rgb_image)


# Generates empty images to verifi that YOLO / other model is being trained reasonably
def generate_zero_input(img_path, output_folder):
    output_folder.mkdir(exist_ok=True)

    images_glob = img_path.glob("*.jpg")

    for file_path in images_glob:
        img = io.imread(str(file_path))
        new_img = np.zeros(img.shape)

        output_file_path = output_folder / file_path.name
        io.imsave(str(output_file_path), new_img)


# make train/val/test splits
# 20 % test set
# 20 % validation set
# hierarchy
# test
#   images
#   labels
# train
#   train
#      images
#      labels
#   val
#      images
#      labels
def data_split(data_path, archive_path, annotations_path, output_path, train_proportion=0.8, folds_num=5):

    output_path.mkdir()

    image_path = data_path / "images" / "train"
    labels_path = data_path / "labels" / "train"

    # Get a list of slices for negative patients
    negative_patients = load_negative_patients(archive_path)
    negative_ids = []

    # Extract slices, check if the orientation is correct
    for patient in negative_patients:
        full_ids = [cinemri_slice.full_id for cinemri_slice in patient.cinemri_slices]
        negative_ids = negative_ids + full_ids

    # Get a list of slices with annotations
    annotated_slices = load_annotated_slices(annotations_path)
    positive_ids = [cinemri_slice.full_id for cinemri_slice in annotated_slices]

    # read ids in the full dataset
    images_glob = image_path.glob("*.jpg")
    dataset_ids = [file.stem for file in images_glob]

    # filter negative and postive slices lists with a list of slices with annotations
    negative_ids = [full_id for full_id in negative_ids if full_id in dataset_ids]
    positive_ids = [full_id for full_id in positive_ids if full_id in dataset_ids]

    # split positive and negative separately and extract the test set
    random.shuffle(negative_ids)
    random.shuffle(positive_ids)

    # Split negative
    train_size_neg = round(len(negative_ids) * train_proportion)
    train_negative = negative_ids[:train_size_neg]
    test_negative = negative_ids[train_size_neg:]

    # Split positive
    train_size_pos = round(len(positive_ids) * train_proportion)
    train_positive = positive_ids[:train_size_pos]
    test_positive = positive_ids[train_size_pos:]

    split_dict = {}

    # Save test set
    test_ids = test_negative + test_positive
    split_dict["test"] = test_ids

    test_set_path = output_path / "test"
    test_set_path.mkdir()

    test_images_path = test_set_path / "images"
    test_images_path.mkdir()

    test_labels_path = test_set_path / "labels"
    test_labels_path.mkdir()

    for test_id in test_ids:
        shutil.copy(image_path / (test_id + ".jpg"), test_images_path)
        label_path = labels_path / (test_id + ".txt")
        if label_path.exists():
            shutil.copy(label_path, test_labels_path)

    # Split training set into folds
    kf = KFold(n_splits=folds_num)
    folds = []

    folds_train = []
    folds_val = []
    train_negative = np.array(train_negative)
    # Handle negatives
    for train_index, val_index in kf.split(train_negative):
        fold_train_ids = train_negative[train_index]
        fold_val_ids = train_negative[val_index]

        folds_train.append(fold_train_ids)
        folds_val.append(fold_val_ids)

    train_positive = np.array(train_positive)
    # Handle positives
    for index, (train_index, val_index) in enumerate(kf.split(train_positive)):
        fold_train_ids = train_positive[train_index]
        fold_val_ids = train_positive[val_index]

        folds_train[index] = np.concatenate((folds_train[index], fold_train_ids))
        folds_val[index] = np.concatenate((folds_val[index], fold_val_ids))

    # Save folds
    index = 1
    train_set_path = output_path / "train"
    train_set_path.mkdir()
    for train_ids, val_ids in zip(folds_train, folds_val):
        fold = {"train": train_ids.tolist(), "val": val_ids.tolist()}
        folds.append(fold)

        # Create fold folder
        fold_dir = train_set_path / "fold{}".format(index)
        fold_dir.mkdir(exist_ok=True)

        # Create images folder
        fold_images_path = fold_dir / "images"
        fold_images_path.mkdir()

        # Create labels folder
        fold_labels_path = fold_dir / "labels"
        fold_labels_path.mkdir()

        # Create train folders
        train_images_path = fold_images_path / "train"
        train_images_path.mkdir()

        train_labels_path = fold_labels_path / "train"
        train_labels_path.mkdir()

        # Copy files
        for train_id in train_ids:
            shutil.copy(image_path / (train_id + ".jpg"), train_images_path)
            label_path = labels_path / (train_id + ".txt")
            if label_path.exists():
                shutil.copy(label_path, train_labels_path)

        # Create validation folders
        val_images_path = fold_images_path / "val"
        val_images_path.mkdir()

        val_labels_path = fold_labels_path / "val"
        val_labels_path.mkdir()

        # Copy files
        for val_id in val_ids:
            shutil.copy(image_path / (val_id + ".jpg"), val_images_path)
            label_path = labels_path / (val_id + ".txt")
            if label_path.exists():
                shutil.copy(label_path, val_labels_path)

        index += 1

    split_dict["train"] = folds
    split_file_path = output_path / "detection_split.json"

    with open(split_file_path, "w") as f:
        json.dump(split_dict, f)


if __name__ == '__main__':
    np.random.seed(99)
    random.seed(99)

    archive_path = Path(ARCHIVE_PATH)
    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    inspexp_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    images_path = archive_path / IMAGES_FOLDER

    masks_path = archive_path / "full_segmentation" / "merged_segmentation"
    visceral_slide_path = Path("../../data/visceral_slide_all/visceral_slide")
    visualization_path = archive_path / "deformation_vis"

    detection_input_contour_path = archive_path / "detection_input_contour"
    detection_input_contour_labels_path = detection_input_contour_path / "labels"

    detection_input_whole_path = archive_path / "detection_input_whole"
    detection_input_whole_labels_path = detection_input_contour_path / "labels"

    detection_contour_npy_path = archive_path / "detection_input_whole" / "full"
    detection_contour_jpg_path = archive_path / "detection_input_whole" / "full_jpg1"

    adhesions_input = Path("../../data/adhesions_local")
    output_path = Path("../../data/adhesions_local_cv")

    data_split(adhesions_input, archive_path, annotations_path, output_path)


    
    #npy_to_jpg(detection_contour_npy_path, detection_contour_jpg_path)

    #contour_types = [AdhesionType.anteriorWall.value, AdhesionType.abdominalCavityContour.value]
    """
    extract_detection_input_subset(detection_input_whole_path,
                                   detection_input_contour_path,
                                   annotations_path,
                                   contour_types)
    """
    #annotations_to_yolov5(annotations_path, detection_input_contour_labels_path, contour_types)
    
    #extract_negative_samples(archive_path, output_path)
    #save_visualization(annotations_path, inspexp_path, images_path, masks_path, output_path=visualization_path)
    #visualize_deformation_and_annotations(annotations_path, inspexp_path, images_path, masks_path, index=2, output_path=visualization_path)
    #extract_annotated_slices(annotations_path, inspexp_path, images_path, masks_path, output_path)