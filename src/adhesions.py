import json
import subprocess
from enum import Enum, unique
from pathlib import Path
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import *
from cinemri.config import ARCHIVE_PATH
from cinemri.contour import AbdominalContourPart, get_contour, Contour
from cinemri.definitions import CineMRISlice
from visceral_slide import VisceralSlideDetector
from cinemri.utils import get_patients
from utils import interval_overlap, load_visceral_slides, get_inspexp_frames
from visceral_slide_pipeline import load_visceral_slide
from contour import get_adhesions_prior_coords

# Folder to save visualized annotations
ANNOTATIONS_VIS_FOLDER = "vis_annotations"

@unique
class AnnotationType(Enum):
    bounding_box = 1
    positive = 2
    negative = 3

@unique
class AdhesionType(Enum):
    unset = 0
    anteriorWall = 1
    pelvis = 2
    inside = 3

@unique
class PatientsType(Enum):
    positive = 0
    negative = 1
    all = 2


class Adhesion:
    """
    An object representing an adhesion
    Attributes
    ----------
       origin_x, origin_y : float
          Origin of a bounding box by x and y axes
       width, height : float
          Width and height of a bounding box
       type : AdhesionType
          A type of adhesion w.r.t. its location
    """


    def __init__(self, bounding_box, type=AdhesionType.unset):
        """
        Parameters
        ----------
        bounding_box : list of float
           A list containing origin of a bounding box and its width and height [origin_x, origin_y, width, height]
        type : AdhesionType
           A type of adhesion w.r.t. its location
        """
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
        """
        Checks whether width and height of an adhesion bounding box is less than the minimum values
        and if it is, sets the size of a bounding box to the specified minimum. The center of the bounding box is kept
        and the origin is shifted accordingly
        Parameters
        ----------
        min_width, min_height : float
           A minimum width and height required for a bounding box
        """

        center = self.center

        if self.width < min_width:
            self.origin_x = max(center[0] - round(min_width / 2), 0)
            self.width = min_width

        if self.height < min_height:
            self.origin_y = max(center[1] - round(min_height / 2), 0)
            self.height = min_height

    def adjust_center(self, new_center):
        delta_x = self.center[0] - new_center[0]
        delta_y = self.center[1] - new_center[1]

        self.origin_x -= delta_x
        self.origin_y -= delta_y

    def contains_point(self, x, y, tolerance=0):

        """ Check if a point belongs to the adhesion

        Parameters
        ----------
        x, y : a point coordinates by x and y axes
        tolerance: extra margin around the bounding box to register a hit
        """
        x_min, x_max = self.origin_x - tolerance, self.max_x + tolerance
        y_min, y_max = self.origin_y - tolerance, self.max_y + tolerance

        return x_min <= x <= x_max and y_min <= y <= y_max

    def intersects_contour(self, contour_x, contour_y, tolerance=0):
        """ Check if a point belongs to the adhesion bounding box

        Parameters
        ----------
        contour_x, contour_y : a list of contour coordinates by x and y axies
        tolerance: extra margin around the bounding box to register a hit

        Returns
        -------
        intersects : bool
           A boolean flag indicating whether a bounding box intersects a contour
        """

        intersects = False
        for x, y in zip(contour_x, contour_y):
            intersects = self.contains_point(x, y, tolerance=tolerance)
            if intersects:
                break

        return intersects

    def contour_point_closes_to_center(self, contour_coords):
        diff = (contour_coords[:, 0] - self.center[0])**2 + (contour_coords[:, 1] - self.center[1])**2
        closest_point_index = np.argmin(diff)
        return contour_coords[closest_point_index]

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
           Calculated intersection over union
        """
        intersect_w = interval_overlap([self.origin_x, self.max_x], [adhesion.origin_x, adhesion.max_x])
        intersect_h = interval_overlap([self.origin_y, self.max_y], [adhesion.origin_y, adhesion.max_y])

        intersect = intersect_w * intersect_h
        union = self.width * self.height + adhesion.width * adhesion.height - intersect

        return float(intersect) / union


class AdhesionAnnotation:
    """
    An object representing an adhesion annotation
    Attributes
    ----------
        slice : CineMRISlice
            A cine-MRI slice for which an annotation is made
        adhesions : list of Adhesion
            A list of adhesions specified in the annotation
    """

    def __init__(self, patient_id, study_id, slice_id, bounding_boxes, types=None):
        """
        Parameters
        ----------
        patient_id, study_id, slice_id : str
           IDs of a patient, study and slices of an annotation
        bounding_boxes : list of float
           A list of bounding boxes of annotated adhesions
        types : list of AdhesionType, optional
           A list of adhesion types corresponding to bouding boxes
        """
        self.slice = CineMRISlice(slice_id, patient_id, study_id)
        if types is None:
            types = [AdhesionType.unset for _ in bounding_boxes]
        self.adhesions = [Adhesion(bounding_box, type) for bounding_box, type in zip(bounding_boxes, types)]

    @property
    def patient_id(self):
        return self.slice.patient_id

    @property
    def study_id(self):
        return self.slice.study_id

    @property
    def slice_id(self):
        return self.slice.id

    @property
    def full_id(self):
        return self.slice.full_id

    def has_adhesion_of_types(self, types):
        """
        Checks whether annotation has at lest one adhesion of any type in a specified list of types
        Parameters
        ----------
        types: list of AdhesionType
           A list of adhesion to check for presence

        Returns
        -------
           has_type : bool
              A boolean flag indicating presence of adhesion of the specified types
        """
        for adhesion in self.adhesions:
            if adhesion.type in types:
                return True

        return False

    def build_path(self, relative_path, extension=".mha"):
        """
        Builds path to a file containing a cine-MRI slice for which annotation is made
        Parameters
        ----------
        relative_path : Path
           A location of a cine-MRI slice
        extension : str
           Cine-MRI slice file extension

        Returns
        -------
        path : Path
           A path to a cine-MRI study

        """
        return self.slice.build_path(relative_path, extension=extension)


# TODO: maybe change to also return slices for which adhesions were not found
def load_annotations(annotations_path,
                     as_dict=False,
                     adhesion_types=[AdhesionType.anteriorWall,
                                     AdhesionType.pelvis,
                                     AdhesionType.inside]):
    """
    Loads adhesion annotations from the metadata file with annotations. Handles two formats of annotations:
    1. adhesions are specified as array of bouding boxes arrays [origin_x, origin_y, width, height]
    2. adhesions are specified as array of dictionaries {"bb": [origin_x, origin_y, width, height], "type" : int}
    Parameters
    ----------
    annotations_path : Path
       A path to the metadata file with annotations
    adhesion_types : list of AdhesionType
       A list of adhesions type to filter adhesion in annotations

    Returns
    -------
    annotations : list of Adhesion
       A list of adhesions annotations with adhesions filtered by types specified in the adhesion_types argument

    """

    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    annotations = {} if as_dict else []
    for patient_id, studies_dict in annotations_dict.items():
        for study_id, slices_dict in studies_dict.items():
            for slice_id, bounding_box_annotations in slices_dict.items():
                # Include bounding boxes which have requested type
                bounding_boxes = []
                types = []
                for bounding_box_annotation in bounding_box_annotations:
                    if type(bounding_box_annotation) is dict:
                        adhesion_type = AdhesionType(bounding_box_annotation["type"])
                        if adhesion_type in adhesion_types:
                            bounding_boxes.append(bounding_box_annotation["bb"])
                            types.append(adhesion_type)
                    else:
                        bounding_boxes.append(bounding_box_annotation)
                        types.append(AdhesionType.unset)

                if len(bounding_boxes) > 0:
                    annotation = AdhesionAnnotation(patient_id, study_id, slice_id, bounding_boxes, types)
                    if as_dict:
                        annotations[annotation.full_id] = annotation
                    else:
                        annotations.append(annotation)

    return annotations


def load_annotated_slices(annotations_path,
                          adhesion_types=[AdhesionType.anteriorWall,
                                          AdhesionType.pelvis,
                                          AdhesionType.inside]):
    """
    Loads annotated slices from the metadata file with annotations. Handles two formats of annotations:
    1. adhesions are specified as array of bouding boxes arrays [origin_x, origin_y, width, height]
    2. adhesions are specified as array of dictionaries {"bb": [origin_x, origin_y, width, height], "type" : int}
    Parameters
    ----------
    annotations_path : Path
       A path to the metadata file with annotations
    adhesion_types : list of AdhesionType
       A list of adhesions type to filter slices

    Returns
    -------
    slices : list of CineMRISlice
       A list of slices which has at least one adhesion of any of types specified in the adhesion_types argument
    """

    annotations = load_annotations(annotations_path, adhesion_types)
    slices = [annotation.slice for annotation in annotations if annotation.has_adhesion_of_types(adhesion_types)]
    return slices


# annotations_path - path to reports with patient level annotations
def load_report_annotations(annotations_path, patients_type=PatientsType.all):
    """
    Loads a list of patient ids positive or negative according to the initial report
    Parameters
    ----------
    annotations_path : Path
       A path to binary patient level annotations from the original report
    patients_type : PatientsType
       A enum case indicating which subset of patients to include: all, positive or negative

    Returns
    -------
    patient_ids : list of str
       A list of ids of patients of the specified type

    """

    with open(annotations_path) as annotations_file:
        reports = json.load(annotations_file)

    if patients_type == PatientsType.all:
        annotation_type_ids = ["0", "1"]
    elif patients_type == PatientsType.negative:
        annotation_type_ids = ["1"]
    else:
        annotation_type_ids = ["0"]

    patient_ids = []
    for report in reports:
        if report["normal"] in annotation_type_ids:
            patient_ids.append(report["id"])

    return patient_ids


def load_patient_ids_reader_study(annotations_path, patients_type=PatientsType.all):
    """
    Extract ids of patients that were included into the reader study
    Parameters
    ----------
    annotations_path : Path
       A path to the metadata file with annotations from the reader study
    patients_type : PatientsType
       A enum case indicating which subset of patients to include: all, positive or negative
    Returns
    -------
    patient_ids : list of str
       A list of ids of patients of the specified type that were included into the reader study
    """
    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    # If we want to extract all patients, just return all keys
    if patients_type == PatientsType.all:
        return list(annotations_dict.keys())

    # First, calculate how many annotations a patient has
    patients = []
    for patient_id, studies_dict in annotations_dict.items():
        annotated_slices_count = 0
        for study_id, slices_dict in studies_dict.items():
            for slice_id, bounding_boxes in slices_dict.items():
                if len(bounding_boxes) > 0:
                    annotated_slices_count += 1

        patients.append((patient_id, annotated_slices_count))

    # Then filter out positive or negative patients
    if patients_type == PatientsType.positive:
        patient_ids = [p for p, n in patients if n > 0]
    else:
        patient_ids = [p for p, n in patients if n == 0]

    return patient_ids


# Those who was included into the reader study and do not have bounding boxes
# And those who was not included into the reader study and is negative in the report data
def load_patients_of_type(images_path, annotations_type_path, annotation_type):
    """
    Loads negative patients filtered by the most accurate annotation available
    Parameters
    ----------
    images_path : Path
       images_path : a path to images folder to look for patients
    annotations_type_path : Path
       A path to the metadata file containing annotation type info {"patient_id":AnnotationType.value}
    annotation_type : AnnotationType
       An annotation type to filter by

    Returns
    -------
    filtered_patients : list of Patient
       A lits of patients having the most accurate annotation type annotation_type
    """

    with open(annotations_type_path) as annotations_type_file:
        annotations_type = json.load(annotations_type_file)

    all_patients = get_patients(images_path)
    type_patients_ids = [patient_id for patient_id, ann_type in annotations_type.items() if
                         AnnotationType(ann_type) == annotation_type]

    filtered_patients = [p for p in all_patients if p.id in type_patients_ids]
    return filtered_patients




def extract_annotations_metadata(images_path,
                                 metadata_path,
                                 bb_annotations_path,
                                 reader_annotations_path,
                                 annotations_type_file=ANNOTATIONS_TYPE_FILE):
    """
    Extract annotation type for each patient and saves as a metadata file.
    Possible annotation types are in AnnotationType enum. A new metadata file name is ANNOTATIONS_TYPE_FILE
    and it is saved inside the metadata folder of an archive.

    Parameters
    ----------
    images_path : Path
       A path to a folder with cine-MRI images
    metadata_path : str
       A path to a folder with metadata
    bb_annotations_path : Path
       A path to the metadata file with bounding box annotations
    reader_annotations_path : Path
       A path to the metadata file with binary patient level annotations
    annotations_type_file : str
       A name of a new file with annotations type
    """
    all_patients = get_patients(images_path)
    all_patient_ids = [p.id for p in all_patients]

    # Extract ids of patients selected for reader study: all, positive and negative
    reader_study_patient_ids = load_patient_ids_reader_study(bb_annotations_path)
    bb_patient_ids = load_patient_ids_reader_study(bb_annotations_path, PatientsType.positive)
    reader_study_negative = load_patient_ids_reader_study(bb_annotations_path, PatientsType.negative)

    # Extract patients which were not included into the reader study
    report_only_patient_ids = list(set(all_patient_ids).difference(set(reader_study_patient_ids)))
    # and are negative according to the original report
    report_negative_patient_ids = load_report_annotations(reader_annotations_path, PatientsType.negative)
    report_only_negative = list(set(report_only_patient_ids) & set(report_negative_patient_ids))

    # Get list of negative patients id according to both reader study and the report
    negative_ids = reader_study_negative + report_only_negative

    # For each patient id specify the most accurate available annotation type
    annotations_type_dict = {}
    for patient_id in all_patient_ids:
        if patient_id in bb_patient_ids:
            annotation_type = AnnotationType.bounding_box
        elif patient_id in negative_ids:
            annotation_type = AnnotationType.negative
        else:
            annotation_type = AnnotationType.positive

        annotations_type_dict[patient_id] = annotation_type.value

    # Save as a new metadata file
    file_path = metadata_path / annotations_type_file
    with open(file_path, "w") as file:
        json.dump(annotations_type_dict, file)


# To have information about adhesion type - v
def extract_adhesions_metadata(annotations_path, full_segmentation_path, metadata_path, expanded_annotations_file):
    """
    Adds type of an adhesion to annotations. Possible types are listed in AdhesionType enum
    Parameters
    ----------
    annotations_path : Path
       A path to annotations metadata file
    full_segmentation_path : Path
       A path to a folder containing full segmentation for cine-MRI studies
    metadata_path : Path
       A path to a metadata folder
    expanded_annotations_file : str
       A file name to save annotations with the added type
    """
    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    annotations_dict_expanded = {}
    for patient_id, studies_dict in annotations_dict.items():
        studies_dict_expanded = {}
        for study_id, slices_dict in studies_dict.items():
            slices_dict_expanded = {}
            for slice_id, bounding_boxes in slices_dict.items():
                segmentation_path = full_segmentation_path / patient_id / study_id / (slice_id + ".mha")
                # Load predicted segmentation for the whole study
                if segmentation_path.exists():
                    slice_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(segmentation_path)))

                adhesions_array = []
                for bounding_box in bounding_boxes:
                    adhesion = Adhesion(bounding_box)
                    intersect_anterior_wall = False
                    intersect_contour = False
                    # For all frames in the slice check if any intersects with adhesion annotation
                    for mask in slice_mask:
                        contour = Contour.from_mask(mask)
                        x_anterior, y_anterior = contour.get_abdominal_contour_part(AbdominalContourPart.anterior_wall)
                        intersect_anterior_wall = adhesion.intersects_contour(x_anterior, y_anterior)
                        if intersect_anterior_wall:
                            intersect_contour = True
                            break

                        intersect_contour |= adhesion.intersects_contour(contour.x, contour.y)

                    if intersect_anterior_wall:
                        adhesion_type = AdhesionType.anteriorWall
                    elif intersect_contour:
                        adhesion_type = AdhesionType.pelvis
                    else:
                        adhesion_type = AdhesionType.inside

                    adhesion_dict = {"bb": bounding_box, "type": adhesion_type.value}
                    adhesions_array.append(adhesion_dict)

                slices_dict_expanded[slice_id] = adhesions_array

            studies_dict_expanded[study_id] = slices_dict_expanded

        annotations_dict_expanded[patient_id] = studies_dict_expanded

    annotations_expanded_path = metadata_path / expanded_annotations_file

    with open(annotations_expanded_path, "w") as file:
        json.dump(annotations_dict_expanded, file)


# TODO: probably remove
def show_annotation(annotation, images_path):
    slice_path = annotation.build_path(images_path)
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


def show_vs_with_annotation(x, y, visceral_slide, frame, annotation=None, normalize=False, title=None, file_name=None):
    """
    Plots absolute value of visceral slide normalized by the absolute maximum together with adhesions annotations
    over the frame of a cine-MRI slice for which visceral slide was computed and saves to a file
    if file_name argument is specified. Otherwise shows the plot
    Parameters
    ----------
    x, y : list of int
       Coordinates of abdominal cavity contour
    visceral_slide
       Visceral slide at abdominal cavity contour
    annotation : AdhesionAnnotation
       Annotation for a slice
    frame : ndarrdy
       A frame for which visceral slide was computed
    title : str, optional
       A title of a plot
    save_file_name : str, optional
       A file name to save the plot
    """

    x_prior, y_prior = get_adhesions_prior_coords(x, y)

    coords = np.column_stack((x, y)).tolist()
    prior_coords = np.column_stack((x_prior, y_prior)).tolist()
    prior_inds = [ind for ind, coord in enumerate(coords) if coord in prior_coords]

    x = x[prior_inds]
    y = y[prior_inds]
    slide_vis = visceral_slide[prior_inds]

    slide_vis = slide_vis / np.max(slide_vis) if normalize else slide_vis

    plt.figure()
    plt.imshow(frame, cmap="gray")
    plt.scatter(x, y, s=5, c=slide_vis, cmap="jet")
    ax = plt.gca()
    if annotation:
        for adhesion in annotation.adhesions:
            adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(adhesion_rect)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.colorbar()
    if file_name:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def annotate_mha(annotations, images_path, target_path):
    """
    Adds bounding box annotation to annotated cine-MRI slices and save in .mha format
    Parameters
    ----------
    annotations : list of AdhesionAnnotation
       A list of annotations to visualize
    images_path : Path
       A path to the image folder in cine-MRI archive
    target_path : Path
       A Path to save annotated slices
    """

    target_path.mkdir(exist_ok=True)

    for annotation in annotations:
        # extract cine-MRI slice
        slice_path = annotation.build_path(images_path)
        slice_image = sitk.ReadImage(str(slice_path))
        slice_series = sitk.GetArrayFromImage(slice_image)

        vis_path = annotation.build_path(target_path, extension="")
        vis_path.mkdir(exist_ok=True)

        # Display bounding box annotations on each frame
        annotated_slice_series = np.zeros(slice_series.shape)
        box_intensity = int(slice_series.max()) + 100
        for frame_id, frame in enumerate(slice_series):
            for adhesion in annotation.adhesions:
                x1, y1 = adhesion.origin_x, adhesion.origin_y
                x2, y2 = adhesion.origin_x + adhesion.width, adhesion.origin_y + adhesion.height
                cv2.rectangle(frame, (x1, y1), (x2, y2), (box_intensity, 0, 0), 1)
            annotated_slice_series[frame_id, ...] = frame

        # Save cine-MRI slice image with visualized annotations and kept metadata
        annotated_slice = sitk.GetImageFromArray(annotated_slice_series)
        annotated_slice.CopyInformation(slice_image)
        patient_id = slice_image.GetMetaData("PatientID")
        study_id = slice_image.GetMetaData("StudyInstanceUID")
        series_id = slice_image.GetMetaData("SeriesInstanceUID")
        annotated_slice.SetMetaData("PatientID", patient_id)
        annotated_slice.SetMetaData("StudyInstanceUID", study_id)
        annotated_slice.SetMetaData("SeriesInstanceUID", series_id)

        annotated_slice_path = vis_path / "series.mha"
        sitk.WriteImage(annotated_slice, str(annotated_slice_path))


def save_annotated_gifs(annotations, image_path, target_path):
    """
    Saves annotations visualized on the corresponding slices as separate frames in .png files and a gif
    Parameters
    ----------
    annotations : list of AdhesionAnnotation
       A list of annotations to visualize
    images_path : Path
       A path to the image folder in cine-MRI archive
    target_path : Path
       A Path to save visualized annotations
    """
    target_path.mkdir(exist_ok=True)

    for annotation in annotations:
        # extract cine-MRI slice
        slice_path = annotation.build_path(image_path)
        slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

        vis_path = annotation.build_path(target_path, extension="")
        vis_path.mkdir(exist_ok=True)

        # Make a separate folder for .png files
        png_path = vis_path / "pngs"
        png_path.mkdir(exist_ok=True)

        # Display an annotation over each frame and save as a .png file
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

        # Combine .png files in a gif and save
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


def vis_annotation_and_saved_vs(annotations_path,
                                images_path,
                                inspexp_file_path,
                                visceral_slide_path,
                                output_path,
                                save=True):
    """
    Visualises saved visceral slide for all annotated slices and saves as .png images is save flag is True
    Parameters
    ----------
    annotations_path, images_path, inspexp_file_path, visceral_slide_path : Path
       Paths to annotations, images, inspiration/expiration and saved visceral slide files
    output_path : Path
       A Path to save visualised annotation and visceral slide over a frame for which visceral slide is computed
    save : bool
       A boolean flag indicating whether to save or show visualisation
    """
    output_path.mkdir(exist_ok=True)
    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    for annotation in annotations:
        visceral_slide_results_path = annotation.build_path(visceral_slide_path, extension="")
        if visceral_slide_results_path.exists():
            # Load the computed visceral slide
            x, y, visceral_slide = load_visceral_slide(visceral_slide_results_path)

            try:
                insp_frame, _ = get_inspexp_frames(annotation.slice, inspexp_data, images_path)
            except:
                print("Missing insp/exp data for the patient {}, study {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.study_id,
                                                                                           annotation.slice_id))

            # Visualise
            annotated_visc_slide_path = output_path / (annotation.full_id + ".png") if save else None
            show_vs_with_annotation(x, y, visceral_slide, insp_frame, annotation, file_name=annotated_visc_slide_path)


def vis_annotation_and_computed_vs(annotations_path,
                                   images_path,
                                   masks_path,
                                   inspexp_file_path,
                                   output_path,
                                   save=True):
    """
    Computes and visualise visceral slide for all annotated slices and saves as .png images is save flag is True
    Parameters
    ----------
    annotations_path, images_path, masks_path, inspexp_file_path : Path
        Paths to annotations, images, masks and inspiration/expiration files
    output_path : Path
        A Path to save visualised annotation and visceral slide over a frame for which visceral slide is computed
    save : bool
        A boolean flag indicating whether to save or show visualisation
    """
    output_path.mkdir(exist_ok=True)

    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    for annotation in annotations:

        # Load inspiration and expiration frames and masks
        try:
            insp_frame, exp_frame = get_inspexp_frames(annotation.slice, inspexp_data, images_path)
            insp_mask, exp_mask = get_inspexp_frames(annotation.slice, inspexp_data, masks_path)
        except:
            print("Missing insp/exp data for the patient {}, study {}, slice {}".format(annotation.patient_id,
                                                                                       annotation.study_id,
                                                                                       annotation.slice_id))

        # Compute visceral slide
        x, y, visceral_slide = VisceralSlideDetector().get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask)
        # Visualise
        annotated_visc_slide_path = output_path / (annotation.full_id + ".png") if save else None
        show_vs_with_annotation(x, y, visceral_slide, insp_frame, annotation, file_name=annotated_visc_slide_path)

# TODO: restore all method here
# def vis_cumulative_vs_for_annotations():

# TODO: fix all usage
def vis_annotation_on_cumulative_vs(visceral_slide_path,
                                    images_path,
                                    annotations_path,
                                    output_path,
                                    adhesion_types=[AdhesionType.anteriorWall,
                                                    AdhesionType.pelvis,
                                                    AdhesionType.inside],
                                    save=True):
    """
    Visualises computed cumulative visceral slide for all annotated slices and saves as .png images is save flag is True
    Parameters
    ----------
    visceral_slide_path, images_path, annotations_path : Path
        Paths to saved visceral slide, cine-MRI studies and annotations
    output_path : Path
        A Path to save visualised annotation and visceral slide over a frame for which visceral slide is computed
    save : bool
        A boolean flag indicating whether to save or show visualisation
    """
    output_path.mkdir(exist_ok=True)

    # load visceral slide
    visceral_slides = load_visceral_slides(visceral_slide_path)
    # load annotations
    annotations_dict = load_annotations(annotations_path, True, adhesion_types)

    # For each vs find annotation is exists and plot
    for visceral_slide in visceral_slides:
        # Load frame that corresponds to the cumulative VS
        slice_path = visceral_slide.build_path(images_path)
        slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
        # Cumulative VS corresponds to the one before the last one frame
        frame = slice[-2]
        # Load annotation if exists
        annotation = annotations_dict[visceral_slide.full_id] if visceral_slide.full_id in annotations_dict else None
        # Visualise
        annotated_visc_slide_path = output_path / (visceral_slide.full_id + ".png") if save else None
        show_vs_with_annotation(visceral_slide.x, visceral_slide.y, visceral_slide.values,
                                frame, annotation, file_name=annotated_visc_slide_path)


def annotations_statistics(expanded_annotations_path):
    """
    Prints statistics of annotations with bounding boxes
    Parameters
    ----------
    expanded_annotations_path : Path
       A path to a metadata file with bounding boxes annotations and added adhesion type
    """

    # load annotations
    annotations = load_annotations(expanded_annotations_path)
    annotations_num = len(annotations)
    print("Number of annotations: {}".format(annotations_num))

    # load slices with visceral slide and annotations
    annotations_to_front_wall = 0
    annotations_to_contour = 0
    annotations_inside = 0
    adhesions_num = 0

    for annotation in annotations:
        adhesions_num += len(annotation.adhesions)

        for adhesion in annotation.adhesions:
            if adhesion.type == AdhesionType.anteriorWall:
                annotations_to_front_wall += 1
            elif adhesion.type == AdhesionType.pelvis:
                annotations_to_contour += 1
            else:
                annotations_inside += 1

    print("Number of adhesions in annotations: {}".format(adhesions_num))

    print("{} of {} are adjacent to the front abdominal wall".format(annotations_to_front_wall, adhesions_num))
    print("{} of {} are adjacent to the abdominal cavity contour".format(annotations_to_contour, adhesions_num))
    print("{} of {} are inside the abdominal cavity".format(annotations_inside, adhesions_num))


def test_cavity_part_detection(annotations_path, images_path, inspexp_file_path, visceral_slide_path, target_path, type=AbdominalContourPart.anterior_wall):
    """
    Visualises detection of the specified abdominal cavity part for all annotated slices
    Parameters
    ----------
    annotations_path, images_path, inspexp_file_path, visceral_slide_path : Path
       Paths to annotations, images, inspiration/expiration data and the computed visceral slide
    target_path : Path
       A path where to save visualisation
    type : AbdominalContourPart
       A type of abdominal cavity part to detect
    """

    target_path.mkdir(exist_ok=True)

    # load annotations
    annotations = load_annotations(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    for annotation in annotations:
        visceral_slide_results_path = annotation.build_path(visceral_slide_path, extension="")
        if visceral_slide_results_path.exists():
            # Load the abdominal cavity contour from the computed visxeral slide info
            x, y, _ = load_visceral_slide(visceral_slide_results_path)

            # Load the inspiration frame (visceral slide is computed for the inspiration frame)
            try:
                insp_frame, _ = get_inspexp_frames(annotation.slice, inspexp_data, images_path)
            except:
                print("Missing insp/exp data for the patient {}, study {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.study_id,
                                                                                           annotation.slice_id))

            if annotation.full_id == "ANON4SV2RE1ET_1.2.752.24.7.621449243.4474616_1.3.12.2.1107.5.2.30.26380.2019060311155223544425245.0.0.0":
                print("a")

            verify_abdominal_wall(x, y, insp_frame, annotation.full_id, target_path, type)


def verify_abdominal_wall(x, y, frame, slice_id, target_path, type=AbdominalContourPart.anterior_wall):
    """
    Allows to visually evaluate the detection of the specified abdominal cavity part by plotting
    abdominal cavity contour over the frame and the detected cavity part over the frame next to each other
    Parameters
    ----------
    x, y : list of int
       Coordinates of abdominal cavity contour
    frame : ndarray
       Frame for which the abdominal cavity contour is computed
    slice_id : str
       A full id of the slice
    target_path : Path
       A path where to save the images
    type : AbdominalContourPart
       A type of the abdominal cavity contour to verify
    """
    x_abdominal_wall, y_abdominal_wall = get_adhesions_prior_coords(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(frame, cmap="gray")
    ax.scatter(x_abdominal_wall, y_abdominal_wall, s=4, color="r")
    plt.axis('off')
    #ax.scatter(x, y, s=4, color="r")


    """
        if type == AbdominalContourPart.top:
        x_abdominal_wall, y_abdominal_wall = get_abdominal_contour_top(x, y)
    else:
        x_abdominal_wall, y_abdominal_wall = get_abdominal_wall_coord(x, y, type)
    """

    """
    ax = fig.add_subplot(122)
    plt.imshow(frame, cmap="gray")
    plt.axis('off')
    ax.scatter(x_abdominal_wall, y_abdominal_wall, s=4, color="r")
    plt.axis('off')
    """

    plt.savefig(target_path / (slice_id + ".png"), bbox_inches='tight', pad_inches=0)
    plt.close()


def test():
    archive_path = Path(ARCHIVE_PATH)
    metadata_path = archive_path / METADATA_FOLDER
    visceral_slide_path = Path("../../data/visceral_slide_all/visceral_slide")
    full_segmentation_path = archive_path / FULL_SEGMENTATION_FOLDER / "merged_segmentation"
    full_segmentation_path = Path(DETECTION_PATH) / "full_segmentation"
    bb_annotation_path = metadata_path / BB_ANNOTATIONS_FILE
    images_path = archive_path / IMAGES_FOLDER


    # detection_path = Path("../../data/cinemri_mha/detection_new") / IMAGES_FOLDER
    ie_file = metadata_path / INSPEXP_FILE_NAME
    cumulative_vs_path = Path("../../data/vs_cum/cumulative_vs_contour_reg_det_full_df")
    # output_path = Path("../../data/visualization/visceral_slide/cumulative_vs_contour_reg_det_full_df")
    detection_path = Path(DETECTION_PATH) / IMAGES_FOLDER / TRAIN_FOLDER
    cumulative_vs_path = Path(DETECTION_PATH) / "output_folder_cum"
    output_path = Path(DETECTION_PATH) / "visualization/cum_vs_warp_contour_norm_avg_rest1"
    bb_expanded_annotation_path = Path(DETECTION_PATH) / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    bb_expanded_annotation_path = Path(DETECTION_PATH) / METADATA_FOLDER / "annotations_expanded1.json"

    extract_adhesions_metadata(bb_annotation_path, full_segmentation_path, metadata_path, bb_expanded_annotation_path)

    """
    vis_annotation_on_cumulative_vs(cumulative_vs_path, detection_path, bb_expanded_annotation_path, output_path,
                                    adhesion_types=[AdhesionType.anteriorWall,
                                                    AdhesionType.pelvis],
                                    save=True)
    """

    #test_cavity_part_detection(bb_expanded_annotation_path, images_path, ie_file, cumulative_vs_path, Path("posterior"), AbdominalContourPart.posterior_wall)
    #test_cavity_part_detection(bb_expanded_annotation_path, images_path, ie_file, visceral_slide_path,
     #                          Path("anterior"), AbdominalContourPart.anterior_wall)
    #test_cavity_part_detection(bb_expanded_annotation_path, images_path, ie_file, visceral_slide_path,
     #                          Path("prior"), AbdominalContourPart.posterior_wall)


if __name__ == '__main__':
    test()
