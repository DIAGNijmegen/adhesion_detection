import json
import pickle
import subprocess
from pathlib import Path
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, SEPARATOR, PATIENT_ANNOTATIONS_FILE_NAME, \
    BB_ANNOTATIONS_FILE, BB_ANNOTATIONS_EXPANDED_FILE, ANNOTATIONS_TYPE_FILE
from cinemri.config import ARCHIVE_PATH
from cinemri.contour import get_contour, get_abdominal_wall_coord, AbdominalContourPart
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np
from cinemri.utils import CineMRISlice
from enum import Enum, unique
from visceral_slide import VisceralSlideDetector
from cinemri.utils import get_patients
from utils import interval_overlap

# Folder to save visualized annotations
ANNOTATIONS_VIS_FOLDER = "vis_annotations"

# TODO: clean up the code for detection of negative patients and statistics of reader study/ report
# TODO: this file requires serious clean up

@unique
class AnnotationType(Enum):
    bounding_box = 1
    positive = 2
    negative = 3

@unique
class AdhesionType(Enum):
    unset = 0
    anteriorWall = 1
    abdominalCavityContour = 2
    inside = 3


class Adhesion:

    def __init__(self, bounding_box, type=AdhesionType.unset):
        self.origin_x = bounding_box[0]
        self.origin_y = bounding_box[1]
        self.width = bounding_box[2]
        self.height = bounding_box[3]
        self.type = type

    @property
    def center(self):
        return self.origin_x + round(self.width / 2), self.origin_y + round(self.height / 2)

    @property
    def max_x(self):
        return self.origin_x + self.width

    @property
    def max_y(self):
        return self.origin_y + self.height

    def adjust_size(self, min_width=15, min_height=15):

        center = self.center

        if self.width < min_width:
            self.origin_x = max(center[0] - round(min_width / 2), 0)
            self.width = min_width

        if self.height < min_height:
            self.origin_y = max(center[1] - round(min_height / 2), 0)
            self.height = min_height


    def contains_point(self, x, y, tolerance=0):

        """ Check if a point belongs to the adhesion

        Parameters
        ----------
        x : a point coordinate on x axis
        y: a point coordinate on y axis
        tolerance: extra margin around the bounding box to register a hit
        """
        x_min, x_max = self.origin_x - tolerance, self.origin_x + self.width + tolerance
        y_min, y_max = self.origin_y - tolerance, self.origin_y + self.height + tolerance

        return x_min <= x <= x_max and y_min <= y <= y_max

    def intersects_contour(self, contour_x, contour_y, tolerance=0):
        """ Check if a point belongs to the adhesion

        Parameters
        ----------
        contour_x : a list of contour coordinates on x axis
        contour_y: a list of contour coordinates on y axis
        tolerance: extra margin around the bounding box to register a hit
        """

        intersects = False
        for x, y in zip(contour_x, contour_y):
            intersects = self.contains_point(x, y, tolerance=tolerance)
            if intersects:
                break

        return intersects

    def iou(self, adhesion):
        """
        Computes intersection over union with another adhesion

        Parameters
        ----------
        adhesion : Adhesion
           An adhesion annotation with which to compute IoU

        Returns
        -------
        iou : float
        """
        intersect_w = interval_overlap([self.origin_x, self.max_x], [adhesion.origin_x, adhesion.max_x])
        intersect_h = interval_overlap([self.origin_y, self.max_y], [adhesion.origin_y, adhesion.max_y])

        intersect = intersect_w * intersect_h
        union = self.width * self.height + adhesion.width * adhesion.height - intersect

        return float(intersect) / union



class AdhesionAnnotation:

    def __init__(self, patient_id, scan_id, slice_id, bounding_boxes, types=None):
        self.slice = CineMRISlice(patient_id, scan_id, slice_id)
        if types is None:
            types = [AdhesionType.unset for _ in bounding_boxes]
        self.adhesions = [Adhesion(bounding_box, type) for bounding_box, type in zip(bounding_boxes, types)]

    @property
    def patient_id(self):
        return self.slice.patient_id

    @property
    def scan_id(self):
        return self.slice.scan_id

    @property
    def slice_id(self):
        return self.slice.slice_id

    @property
    def full_id(self):
        return self.slice.full_id

    def build_path(self, relative_path, extension=".mha"):
        return Path(relative_path) / self.patient_id / self.scan_id / (self.slice_id + extension)


# Assume we have expanded annotation with bb types
def load_annotations(annotations_path,
                     adhesion_types=[AdhesionType.anteriorWall.value,
                                     AdhesionType.abdominalCavityContour.value,
                                     AdhesionType.inside.value]):

    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    annotations = []
    for patient_id, scans_dict in annotations_dict.items():
        for scan_id, slices_dict in scans_dict.items():
            for slice_id, bounding_box_dicts in slices_dict.items():
                # Include bounding boxes which have requested type
                bounding_boxes = []
                types = []
                for bounding_box_dict in bounding_box_dicts:
                    if bounding_box_dict["type"] in adhesion_types:
                        bounding_boxes.append(bounding_box_dict["bb"])
                        types.append(bounding_box_dict["type"])

                if len(bounding_boxes) > 0:
                    annotation = AdhesionAnnotation(patient_id, scan_id, slice_id, bounding_boxes, types)
                    annotations.append(annotation)

    return annotations

# Assume we have expanded annotation with bb types
def load_annotated_slices(annotations_path,
                          adhesion_types=[AdhesionType.anteriorWall.value,
                                          AdhesionType.abdominalCavityContour.value,
                                          AdhesionType.inside.value]):
    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    slices = []
    for patient_id, scans_dict in annotations_dict.items():
        for scan_id, slices_dict in scans_dict.items():
            for slice_id, bounding_box_dicts in slices_dict.items():
                if len(bounding_box_dicts) > 0:
                    # If any of BB annotations for a slice has a requested type, include this slice
                    has_requested_type = False
                    for bounding_box_dict in bounding_box_dicts:
                        if bounding_box_dict["type"] in adhesion_types:
                            has_requested_type = True
                            break

                    if has_requested_type:
                        slice = CineMRISlice(patient_id, scan_id, slice_id)
                        slices.append(slice)

    return slices


# annotations_path - path to reports with patient level annotations
def load_negative_ids_rijnstate(annotations_path):

    with open(annotations_path) as annotations_file:
        reports = json.load(annotations_file)

    patient_ids = []
    for report in reports:
        # If patient is healthy append the id to array
        if report["normal"] == "1":
            patient_ids.append(report["id"])

    return patient_ids


# Extract ids of patients that were included into the reader study
def load_patient_ids_reader_study(annotations_path):
    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    return list(annotations_dict.keys())


# annotations_path - path to reports with patient level annotations
def load_patients_with_bb_ids(annotations_path):

    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    # In the dictionary with annotations some patients have an empty array of annotations
    # We need filter them out
    # First, calculate how many annotations a patient has
    patients = []
    for patient_id, scans_dict in annotations_dict.items():
        annotated_slices_count = 0
        for scan_id, slices_dict in scans_dict.items():
            for slice_id, bounding_boxes in slices_dict.items():
                if len(bounding_boxes) > 0:
                    annotated_slices_count += 1

        patients.append((patient_id, annotated_slices_count))

    # Then filter out patients without annotations
    patient_ids = np.array([p for p, n in patients if n > 0])
    return patient_ids


# annotations_path - path to reports with patient level annotations
def load_patients_without_bb_ids(archive_path):

    all_patients = get_patients(archive_path)

    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_FILE
    patients_with_bb_annotations = load_patients_with_bb_ids(annotations_path)

    patients = [p for p in all_patients if p.id not in patients_with_bb_annotations]
    return patients


# Those who was included into the reader study and do not have bounding boxes
# And those who was not included into the reader study and is negative in the report data
def load_negative_patients(archive_path):
    annotations_type_path = archive_path / METADATA_FOLDER / ANNOTATIONS_TYPE_FILE

    with open(annotations_type_path) as annotations_type_file:
        annotations_type = json.load(annotations_type_file)

    negative_patients_ids = [patient_id for patient_id, annotation_type in annotations_type.items() if annotation_type == AnnotationType.negative.value]
    all_patients = get_patients(archive_path)

    negative_patients = [p for p in all_patients if p.id in negative_patients_ids]
    return negative_patients


# To quickly know which patients have bb annotations and which have binary annotations
def extract_annotations_metadata(archive_path):
    all_patients = get_patients(archive_path)
    all_patient_ids = [p.id for p in all_patients]

    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_FILE
    reader_study_patient_ids = load_patient_ids_reader_study(annotations_path)
    bb_patient_ids = load_patients_with_bb_ids(annotations_path)

    reader_study_negative = list(set(reader_study_patient_ids).difference(set(bb_patient_ids)))

    report_only_patient_ids = list(set(all_patient_ids).difference(set(reader_study_patient_ids)))
    patient_annotation_path = archive_path / METADATA_FOLDER / PATIENT_ANNOTATIONS_FILE_NAME
    report_negative_patient_ids = load_negative_ids_rijnstate(patient_annotation_path)
    report_only_negative = list(set(report_only_patient_ids) & set(report_negative_patient_ids))

    negative_ids = reader_study_negative + report_only_negative

    annotations_type_dict = {}
    for patient_id in all_patient_ids:
        if patient_id in bb_patient_ids:
            annotation_type = AnnotationType.bounding_box
        elif patient_id in negative_ids:
            annotation_type = AnnotationType.negative
        else:
            annotation_type = AnnotationType.positive

        annotations_type_dict[patient_id] = annotation_type.value

    file_path = archive_path / METADATA_FOLDER / ANNOTATIONS_TYPE_FILE

    with open(file_path, "w") as file:
        json.dump(annotations_type_dict, file)


# To have information about adhesion type - v
def extract_adhesions_metadata(archive_path, annotations_path, full_segmentation_path):
    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    annotations_dict_expanded = {}
    for patient_id, scans_dict in annotations_dict.items():
        scans_dict_expanded = {}
        for scan_id, slices_dict in scans_dict.items():
            slices_dict_expanded = {}
            for slice_id, bounding_boxes in slices_dict.items():
                segmentation_path = full_segmentation_path / patient_id / scan_id / (slice_id + ".mha")
                if segmentation_path.exists():
                    # Load predicted segmentation for the whole scan
                    slice_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(segmentation_path)))

                adhesions_array = []
                for bounding_box in bounding_boxes:
                    adhesion = Adhesion(bounding_box)
                    intersect_anterior_wall = False
                    intersect_contour = False
                    # For all frames in the slice check if any intersects with adhesion annotation
                    for mask in slice_mask:
                        x, y, _, _ = get_contour(mask)
                        x_anterior, y_anterior = get_abdominal_wall_coord(x, y)
                        intersect_anterior_wall = adhesion.intersects_contour(x_anterior, y_anterior)
                        if intersect_anterior_wall:
                            intersect_contour = True
                            break

                        intersect_contour = intersect_contour or adhesion.intersects_contour(x, y)

                    if intersect_anterior_wall:
                        adhesion_type = AdhesionType.anteriorWall
                    elif intersect_contour:
                        adhesion_type = AdhesionType.abdominalCavityContour
                    else:
                        adhesion_type = AdhesionType.inside

                    adhesion_dict = {"bb": bounding_box, "type": adhesion_type.value}
                    adhesions_array.append(adhesion_dict)

                slices_dict_expanded[slice_id] = adhesions_array

            scans_dict_expanded[scan_id] = slices_dict_expanded

        annotations_dict_expanded[patient_id] = scans_dict_expanded

    annotations_expanded_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE

    with open(annotations_expanded_path, "w") as file:
        json.dump(annotations_dict_expanded, file)


def show_annotation(annotation, archive_path):
    slice_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (annotation.slice_id + ".mha")
    # first just plot
    slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

    for frame in slice:
        plt.figure()
        ax = plt.gca()
        for adhesion in annotation.adhesions:
            adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(adhesion_rect)
        plt.imshow(frame, cmap="gray")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def show_annotation_and_vs(x, y, visceral_slide, annotation, expiration_frame, title=None, save_file_name=None):
    slide_normalized = np.abs(visceral_slide) / np.abs(visceral_slide).max()

    plt.figure()
    plt.imshow(expiration_frame, cmap="gray")
    plt.scatter(x, y, s=5, c=slide_normalized, cmap="jet")
    ax = plt.gca()
    for adhesion in annotation.adhesions:
        adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                  linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(adhesion_rect)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.colorbar()
    if save_file_name:
        plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def annotate_mha(annotations, archive_path):
    vis_annotations_path = archive_path / ANNOTATIONS_VIS_FOLDER
    vis_annotations_path.mkdir(exist_ok=True)

    for annotation in annotations:
        # extract image
        slice_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (
                annotation.slice_id + ".mha")
        slice_image = sitk.ReadImage(str(slice_path))
        slice_series = sitk.GetArrayFromImage(slice_image)

        vis_path = vis_annotations_path / annotation.patient_id / annotation.scan_id / annotation.slice_id
        vis_path.mkdir(parents=True, exist_ok=True)

        annotated_slice_series = np.zeros(slice_series.shape)
        box_intensity = int(slice_series.max()) + 100
        for frame_id, frame in enumerate(slice_series):
            for adhesion in annotation.adhesions:
                x1, y1 = adhesion.origin_x, adhesion.origin_y
                x2, y2 = adhesion.origin_x + adhesion.width, adhesion.origin_y + adhesion.height
                cv2.rectangle(frame, (x1, y1), (x2, y2), (box_intensity, 0, 0), 1)
            annotated_slice_series[frame_id, ...] = frame

        annotated_slice = sitk.GetImageFromArray(annotated_slice_series)
        # keep metadata
        annotated_slice.CopyInformation(slice_image)
        patient_id = slice_image.GetMetaData("PatientID")
        study_id = slice_image.GetMetaData("StudyInstanceUID")
        series_id = slice_image.GetMetaData("SeriesInstanceUID")
        annotated_slice.SetMetaData("PatientID", patient_id)
        annotated_slice.SetMetaData("StudyInstanceUID", study_id)
        annotated_slice.SetMetaData("SeriesInstanceUID", series_id)

        annotated_slice_path = vis_path / "series.mha"
        sitk.WriteImage(annotated_slice, str(annotated_slice_path))


def save_annotated_gifs(annotations, archive_path):
    vis_annotations_path = archive_path / ANNOTATIONS_VIS_FOLDER
    vis_annotations_path.mkdir(exist_ok=True)

    for annotation in annotations:
        # extract image
        slice_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (
                    annotation.slice_id + ".mha")

        slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

        vis_path = vis_annotations_path / annotation.patient_id / annotation.scan_id / annotation.slice_id
        vis_path.mkdir(parents=True, exist_ok=True)

        png_path = vis_path / "pngs"
        png_path.mkdir(exist_ok=True)

        for frame_id, frame in enumerate(slice):
            plt.figure()
            plt.imshow(frame, cmap="gray")

            ax = plt.gca()
            for adhesion in annotation.adhesions:
                adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                          linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(adhesion_rect)


            plt.axis("off")
            annotated_frame_file_path = png_path / (str(frame_id) + ".png")
            plt.savefig(annotated_frame_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        command = [
            "convert",
            "-coalesce",
            "-delay",
            "20",
            "-loop",
            "0",
            str(png_path) + "/*png",
            str(vis_path) + "/" + "annotated_slice.gif",
        ]
        subprocess.run(command)


def vis_annotation_and_vs(archive_path,
                          visceral_slide_path,
                          output_path):
    output_path.mkdir(exist_ok=True)

    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_FILE
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    
    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    for annotation in annotations:
        visceral_slide_results_path = visceral_slide_path / annotation.patient_id / annotation.scan_id / annotation.slice_id
        if visceral_slide_results_path.exists():
            # Load the computed visceral slide
            visceral_slide_file_path = visceral_slide_results_path / "visceral_slide.pkl"
            with open(str(visceral_slide_file_path), "r+b") as file:
                visceral_slide_data = pickle.load(file)
                x, y = visceral_slide_data["x"], visceral_slide_data["y"]
                visceral_slide = visceral_slide_data["slide"]

            try:
                patient_data = inspexp_data[annotation.patient_id]
                scan_data = patient_data[annotation.scan_id]
                exp_frame = scan_data[annotation.slice_id][1]
            except:
                print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.scan_id,
                                                                                           annotation.slice_id))

            # Load the expiration frame (visceral slide is computed for the expiration frame)
            expiration_frame_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (annotation.slice_id + ".mha")
            expiration_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(expiration_frame_path)))[exp_frame]

            slice_id = SEPARATOR.join([annotation.patient_id, annotation.scan_id, annotation.slice_id])
            annotated_visc_slide_path = output_path / (slice_id + ".png")

            show_annotation_and_vs(x, y, visceral_slide, annotation, expiration_frame, save_file_name=annotated_visc_slide_path)


def vis_annotation_and_vs1(archive_path,
                           output_path):
    output_path.mkdir(exist_ok=True)

    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_FILE
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME

    images_path = archive_path / IMAGES_FOLDER
    masks_path = archive_path / "full_segmentation" / "merged_segmentation"

    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    for annotation in annotations:

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
        exp_frame = slice_array[inspexp_frames[1]].astype(np.uint32)

        # load masks
        mask_path = annotation.build_path(masks_path)
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        insp_mask = mask_array[inspexp_frames[0]]
        exp_mask = mask_array[inspexp_frames[1]]

        # registration
        x, y, visceral_slide = VisceralSlideDetector().get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask)

        slice_id = SEPARATOR.join([annotation.patient_id, annotation.scan_id, annotation.slice_id])
        annotated_visc_slide_path = output_path / (slice_id + ".png")

        show_annotation_and_vs(x, y, visceral_slide, annotation, insp_frame, save_file_name=annotated_visc_slide_path)


def load_visceral_slide(visceral_slide_file_path):
    # Load the computed visceral slide
    with open(str(visceral_slide_file_path), "r+b") as file:
        visceral_slide_data = pickle.load(file)
        x, y = visceral_slide_data["x"], visceral_slide_data["y"]
        visceral_slide = visceral_slide_data["slide"]

    return x, y, visceral_slide


def load_expiration_frame(archive_path, inspexp_data, patient_id, scan_id, slice_id):
    patient_data = inspexp_data[patient_id]
    scan_data = patient_data[scan_id]
    exp_frame = scan_data[slice_id][1]

    # Load the expiration frame (visceral slide is computed for the expiration frame)
    expiration_frame_path = archive_path / IMAGES_FOLDER / patient_id / scan_id / (slice_id + ".mha")
    expiration_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(expiration_frame_path)))[exp_frame]

    return expiration_frame


def annotations_statistics(archive_path,
                           full_segmentation_path,
                           visceral_slide_path):

    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_FILE
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME

    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    annotations_to_front_wall = 0
    annotations_to_contour = 0
    adhesions_num = 0
    annotations_num = 0
    annotated_num = 0
    annotated_vs_num = 0
    for annotation in annotations:
        annotated_num += len(annotation.adhesions)
        print("Annotation {}".format(annotations_num))
        segmentation_path = full_segmentation_path / annotation.patient_id / annotation.scan_id / (annotation.slice_id + ".mha")
        if segmentation_path.exists():
            visceral_slide_file_path = visceral_slide_path / annotation.patient_id / annotation.scan_id / annotation.slice_id / "visceral_slide.pkl"
            if visceral_slide_file_path.exists():
                x, y, visceral_slide = load_visceral_slide(visceral_slide_file_path)
            else:
                visceral_slide = None
                print("No visceral slide data for patient {}, scan {}, slice {}".format(annotation.patient_id,
                                                                                        annotation.scan_id,
                                                                                        annotation.slice_id))

            try:
                expiration_frame = load_expiration_frame(archive_path, inspexp_data, annotation.patient_id, annotation.scan_id, annotation.slice_id)
            except:
                expiration_frame = None
                print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.scan_id,
                                                                                           annotation.slice_id))

            # To visually evaluate of intersection detection is correct
            if visceral_slide is not None and expiration_frame is not None:
                if annotations_num == 50:
                    show_annotation_and_vs(x, y, visceral_slide, annotation, expiration_frame,
                                           title="Annotation {}".format(annotations_num))

            # Load predicted segmentation for the whole scan
            slice_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(segmentation_path)))
            # For each adhesion in this annotation check if it is adjacent to the front abdominal wall
            for adhesion in annotation.adhesions:
                intersect_anterior_wall = False
                intersect_contour = False
                # For all frames in the slice check if any intersects with adhesion annotation
                for mask in slice_mask:
                    x, y, _, _ = get_contour(mask)
                    x_anterior, y_anterior = get_abdominal_wall_coord(x, y)
                    intersect_anterior_wall = adhesion.intersects_contour(x_anterior, y_anterior)
                    if intersect_anterior_wall:
                        intersect_contour = True
                        break

                    intersect_contour = intersect_contour or adhesion.intersects_contour(x, y)

                if intersect_anterior_wall:
                    print("This adhesion annotation is adjacent to the front abdominal wall")
                    annotations_to_front_wall += 1
                    annotations_to_contour += 1
                elif intersect_contour:
                    print("This adhesion annotation is adjacent to the abdominal cavity contour")
                    annotations_to_contour += 1

                adhesions_num += 1
            annotated_vs_num += 1
        annotations_num += 1

    print("{} of {} annotated adhesions have information about contour".format(adhesions_num, annotated_num))
    print("{} of {} are adjacent to the front abdominal wall".format(annotations_to_front_wall, adhesions_num))
    print("{} of {} are adjacent to the abdominal cavity contour".format(annotations_to_contour, adhesions_num))


# A function to examine the detected front wall for all annotation for each
# abdominal cavity contour is available at the moment
def test_abdominal_wall_detection(archive_path, visceral_slide_path, target_path, type=AbdominalContourPart.anterior_wall):

    target_path.mkdir(exist_ok=True)

    annotations_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME

    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    for annotation in annotations:
        visceral_slide_results_path = visceral_slide_path / annotation.patient_id / annotation.scan_id / annotation.slice_id
        if visceral_slide_results_path.exists():
            # Load the computed visceral slide
            visceral_slide_file_path = visceral_slide_results_path / "visceral_slide.pkl"
            with open(str(visceral_slide_file_path), "r+b") as file:
                visceral_slide_data = pickle.load(file)
                x, y = visceral_slide_data["x"], visceral_slide_data["y"]

            try:
                patient_data = inspexp_data[annotation.patient_id]
                scan_data = patient_data[annotation.scan_id]
                insp_frame = scan_data[annotation.slice_id][0]
            except:
                print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.scan_id,
                                                                                           annotation.slice_id))

            # Load the inspiration frame (visceral slide is computed for the inspiration frame)
            inspiration_frame_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (
                    annotation.slice_id + ".mha")
            insp_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(inspiration_frame_path)))[insp_frame]

            slice_id = SEPARATOR.join([annotation.patient_id, annotation.scan_id, annotation.slice_id])
            verify_abdominal_wall(x, y, insp_frame, slice_id, target_path, type)



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
       A list of start and end coordinates of all connected regions

    """

    regions = []
    region_start = contour_subset_coords[0]
    coord_prev = region_start
    coords_num = contour_subset_coords.shape[0]
    for index in range(1, coords_num):
        coord_curr = contour_subset_coords[index]
        if abs(coord_curr[1] - coord_prev[1]) > connectivity_threshold:
            regions.append([region_start, coord_prev])
            region_start = coord_curr
        coord_prev = coord_curr

    regions.append([region_start, contour_subset_coords[-1]])

    return regions


def get_abdominal_contour_top(x, y, connectivity_threshold=5):
    """
    Extracts coordinates of the abdominal wall of the specified type from abdominal cavity contour

    Parameters
    ----------
    x, y : list of int
       x-axis and y-axis components of abdominal cavity contour
    type : AbdominalContourPart
       indicates which abdominal wall to detect, anterior or posterior

    connectivity_threshold : int, default=5
       Threshold which indicates the maximum difference in x component allowed between
       the subsequent coordinates of a contour to be considered connected

    Returns
    -------
    x_wall, y_wall : ndarray of int
       x-axis and y-axis components of abdominal wall
    """

    coords = np.column_stack((x, y))
    # Get unique x coordinates
    x_unique = np.unique(x)

    # For each unique y find the x on the specified side of abdominal cavity contour
    y_top = []
    for x_current in x_unique:
        # Get all x values of coordinates which have y = y_current
        ys = [coord[1] for coord in coords if coord[0] == x_current]
        y_top.append(sorted(ys)[0])

    # Unique y coordinates and the corresponding x coordinates
    # should give good approximation of the abdominal wall
    y_top = np.array(y_top)
    top_contour_coords = np.column_stack((x_unique, y_top))

    regions = get_connected_regions(top_contour_coords, connectivity_threshold)

    # Discard regions below the middle y
    y = np.array(y)
    middle_y = (y.max() - y.min()) / 2
    top_regions = []
    for region in regions:
        # Check that min and max y of a region is above middle_y
        if region[0][1] < middle_y and region[1][1] < middle_y:
            top_regions.append(region)

    # Find and return the longest region
    top_regions_len = np.array([(region[1][0] - region[0][0] + 1) for region in top_regions])
    longest_region_ind = np.argmax(top_regions_len)
    longest_region = top_regions[longest_region_ind]

    longest_region_start = longest_region[0]
    longest_region_end = longest_region[1]

    # The contour returned by cinemri.contour.get_contour() always starts from
    # the lowest-left point of the contour, so when we determine the top part
    # we can safely expect that by using the start and end point
    # of the longest top region we will get a connected area from the contour coordinates

    longest_region_start_ind = np.where((coords == longest_region_start).all(axis=1))[0][0]
    longest_region_end_ind = np.where((coords == longest_region_end).all(axis=1))[0][0]
    top_coords = coords[longest_region_start_ind:longest_region_end_ind]

    return top_coords[:, 0], top_coords[:, 1]



# Allows to visually evaluate quality of the detection of front abdominal wall
# By plotting abdominal cavity contour over the frame and the detected front wall over the frame
# next to each other
def verify_abdominal_wall(x, y, exp_frame, slice_id, target_path, type=AbdominalContourPart.anterior_wall):

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.imshow(exp_frame, cmap="gray")
    plt.axis('off')
    ax.scatter(x, y, s=4, color="r")

    x_abdominal_wall, y_abdominal_wall = get_abdominal_contour_top(x, y)
    #x_abdominal_wall, y_abdominal_wall = get_abdominal_wall_coord(x, y, type)

    ax = fig.add_subplot(122)
    plt.imshow(exp_frame, cmap="gray")
    plt.axis('off')
    ax.scatter(x_abdominal_wall, y_abdominal_wall, s=4, color="r")
    plt.axis('off')
    plt.savefig(target_path / (slice_id + ".png"), bbox_inches='tight', pad_inches=0)
    plt.close()


def test():
    archive_path = Path(ARCHIVE_PATH)
    metadata_path = archive_path / METADATA_FOLDER
    visceral_slide_path = Path("../../data/visceral_slide_all/visceral_slide")
    output_path = archive_path / "visceral_slide" / "new_insp_exp"
    full_segmentation_path = archive_path / "full_segmentation" / "merged_segmentation"
    bb_annotation_path = metadata_path / BB_ANNOTATIONS_FILE

    bb_expanded_annotation_path = metadata_path / BB_ANNOTATIONS_EXPANDED_FILE

    #annotations = load_annotations(bb_annotation_path)
    #save_annotated_gifs(annotations, archive_path)
    #vis_annotation_and_vs(archive_path, visceral_slide_path, output_path)
    #vis_annotation_and_vs1(archive_path, output_path)

    test_abdominal_wall_detection(archive_path, visceral_slide_path, Path("contour_top_results_disc_filled"))
    #test_abdominal_wall_detection(archive_path, visceral_slide_path, Path("anterior_wall_results"))
    #test_abdominal_wall_detection(archive_path, visceral_slide_path, Path("posterior_wall_results"), AbdominalContourPart.posterior_wall)
    #annotations_statistics(archive_path, full_segmentation_path, visceral_slide_path)

    #annotations = load_annotations(bb_expanded_annotation_path, adhesion_types=[3])
    #print(len(annotations))

    #extract_adhesions_metadata(archive_path, bb_annotation_path, full_segmentation_path)
    #extract_annotations_metadata(archive_path)

    """
    negative_patients = load_negative_patients(archive_path)
    print("Certainly negative patients")
    print([p.id for p in negative_patients])
    """

    
    # Show randomly sampled annotation
    #annotation = random.choice(annotations)
    #show_annotation(annotation, archive_path)
    #save_annotated_gifs(annotations, archive_path)
    #annotate_mha(annotations, archive_path)

    

if __name__ == '__main__':
    test()
