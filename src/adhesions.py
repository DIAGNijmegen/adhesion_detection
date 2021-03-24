import json
import subprocess
from pathlib import Path
from config import IMAGES_FOLDER, METADATA_FOLDER
from cinemri.config import ARCHIVE_PATH
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import random
import numpy as np

ANNOTATIONS_FILE_NAME = "annotations.json"
ANNOTATIONS_VIS_FOLDER = "vis_annotations"

class Adhesion:

    def __init__(self, bounding_box):
        self.origin_x = bounding_box[0]
        self.origin_y = bounding_box[1]
        self.width = bounding_box[2]
        self.height = bounding_box[3]


class AdhesionAnnotation:

    def __init__(self, patient_id, scan_id, slice_id, bounding_boxes):
        self.patient_id = patient_id
        self.scan_id = scan_id
        self.slice_id = slice_id
        self.adhesions = [Adhesion(bounding_box) for bounding_box in bounding_boxes]


def load_bounding_boxes(annotations_path):
    with open(annotations_path) as annotations_file:
        annotations_dict = json.load(annotations_file)

    annotations = []
    adhesions_num = 0
    for patient_id, scans_dict in annotations_dict.items():
        for scan_id, slices_dict in scans_dict.items():
            for slice_id, bounding_boxes in slices_dict.items():
                if len(bounding_boxes) > 0:
                    adhesions_num += len(bounding_boxes)
                    annotation = AdhesionAnnotation(patient_id, scan_id, slice_id, bounding_boxes)
                    annotations.append(annotation)


    return annotations


def show_annotation(annotation, archive_path):
    slice_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (annotation.slice_id + ".mha")
    # first just plot
    slice = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))

    for frame in slice:
        plt.figure()
        #ax = plt.gca()
        frame_norm = frame / frame.max() * 255
        box_intensity = int(frame.max()) + 100
        frame_norm = frame_norm.astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)
        for adhesion in annotation.adhesions:
            x1 = adhesion.origin_x
            y1 = adhesion.origin_y
            x2 = adhesion.origin_x + adhesion.width
            y2 = adhesion.origin_y + adhesion.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (box_intensity, 0, 0), 1)
            #adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height, linewidth=1, edgecolor='r', facecolor='none')
            #ax.add_patch(adhesion_rect)
        plt.imshow(frame, cmap="gray")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # next save as .mha or even gif


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

        for frame_id, frame in enumerate(slice):
            plt.figure()
            plt.imshow(frame, cmap="gray")
            ax = plt.gca()
            for adhesion in annotation.adhesions:
                adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                          linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(adhesion_rect)
            plt.axis("off")
            annotated_frame_file_path = vis_path / (str(frame_id) + ".png")
            plt.savefig(annotated_frame_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        command = [
            "convert",
            "-coalesce",
            "-delay",
            "20",
            "-loop",
            "0",
            str(vis_path) + "/*png",
            str(vis_path) + "/" + "annotated_slice.gif",
        ]
        subprocess.run(command)


def test():
    archive_path = Path(ARCHIVE_PATH)
    annotations_path = archive_path / METADATA_FOLDER / ANNOTATIONS_FILE_NAME

    annotations = load_bounding_boxes(annotations_path)
    for annotation in annotations:
        print("Patient {}, scan {}, slice {}".format(annotation.patient_id, annotation.scan_id, annotation.slice_id))
        for adhesion in annotation.adhesions:
            print("Adhesion box x: {}, y: {}, width: {}, height: {}".format(adhesion.origin_x, adhesion.origin_y, adhesion.width, adhesion.height))

    # Show randomly sampled annotation
    #annotation = random.choice(annotations)
    #show_annotation(annotation, archive_path)
    #save_annotated_gifs(annotations, archive_path)
    annotate_mha(annotations, archive_path)

    

if __name__ == '__main__':
    test()
