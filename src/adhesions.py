import json
import pickle
import subprocess
from pathlib import Path
from config import IMAGES_FOLDER, METADATA_FOLDER, INSPEXP_FILE_NAME, SEPARATOR
from cinemri.config import ARCHIVE_PATH
from cinemri.contour import _get_tangent_vectors, get_outward_normal_vectors
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np

ANNOTATIONS_FILE_NAME = "annotations.json"
ANNOTATIONS_VIS_FOLDER = "vis_annotations"

class Adhesion:

    def __init__(self, bounding_box):
        self.origin_x = bounding_box[0]
        self.origin_y = bounding_box[1]
        self.width = bounding_box[2]
        self.height = bounding_box[3]

    @property
    def center(self):
        return self.origin_x + round(self.width / 2), self.origin_y + round(self.height / 2)

    def contains_point(self, x, y, tolerance=3):
        x_min, x_max = self.origin_x - tolerance, self.origin_x + self.width + tolerance
        y_min, y_max = self.origin_y - tolerance, self.origin_y + self.height + tolerance

        return x_min <= x <= x_max and y_min <= y <= y_max


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
        ax = plt.gca()
        for adhesion in annotation.adhesions:
            adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(adhesion_rect)
        plt.imshow(frame, cmap="gray")
        plt.axis('off')
        plt.tight_layout()
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


def vis_annotation_and_vs(archive_path,
                          visceral_slide_path,
                          output_path):
    output_path.mkdir(exist_ok=True)

    annotations_path = archive_path / METADATA_FOLDER / ANNOTATIONS_FILE_NAME
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    
    # load annotations
    annotations = load_bounding_boxes(annotations_path)
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
            slide_normalized = np.abs(visceral_slide) / np.max(np.abs(visceral_slide))

            plt.figure()
            plt.imshow(expiration_frame, cmap="gray")
            plt.scatter(x, y, s=5, c=slide_normalized, cmap="jet")
            ax = plt.gca()
            for adhesion in annotation.adhesions:
                adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                          linewidth=1.5, edgecolor='r', facecolor='none')
                ax.add_patch(adhesion_rect)
            plt.axis('off')
            plt.colorbar()
            plt.savefig(annotated_visc_slide_path, bbox_inches='tight', pad_inches=0)
            plt.close()


def annotations_to_abdominal_wall(archive_path,
                                  visceral_slide_path):

    annotations_path = archive_path / METADATA_FOLDER / ANNOTATIONS_FILE_NAME
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME

    # load annotations
    annotations = load_bounding_boxes(annotations_path)
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    # load slices with visceral slide and annotations
    annotations_to_front_wall = 0
    annotations_to_contour = 0
    adhesions_num = 0
    annotations_num = 0
    annotated_num = 0
    for annotation in annotations:
        annotated_num += len(annotation.adhesions)
        print("Annotation {}".format(annotations_num))
        visceral_slide_results_path = visceral_slide_path / annotation.patient_id / annotation.scan_id / annotation.slice_id
        if visceral_slide_results_path.exists():


            # Load the computed visceral slide
            visceral_slide_file_path = visceral_slide_results_path / "visceral_slide.pkl"
            with open(str(visceral_slide_file_path), "r+b") as file:
                visceral_slide_data = pickle.load(file)
                x, y = visceral_slide_data["x"], visceral_slide_data["y"]
                visceral_slide = visceral_slide_data["slide"]

            contour_coord = np.column_stack((x, y))
            anterior_wall_coord = get_anterior_wall_coord(x, y)

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

            slide_normalized = np.abs(visceral_slide) / np.max(np.abs(visceral_slide))
            # To validate detection visually
            if annotations_num > 100:
                plt.figure()
                plt.imshow(expiration_frame, cmap="gray")
                plt.scatter(x, y, s=5, c=slide_normalized, cmap="jet")
                ax = plt.gca()
                for adhesion in annotation.adhesions:
                    adhesion_rect = Rectangle((adhesion.origin_x, adhesion.origin_y), adhesion.width, adhesion.height,
                                              linewidth=1.5, edgecolor='r', facecolor='none')
                    ax.add_patch(adhesion_rect)
                plt.axis("off")
                plt.title("Annotation {}".format(annotations_num))
                plt.colorbar()
                plt.show()

            # For each adhesion in this annotation check if it is adjacent to the front abdominal wall
            for adhesion in annotation.adhesions:
                intersect_anterior_wall = False
                for coord in anterior_wall_coord:
                    intersect_anterior_wall = adhesion.contains_point(coord[0], coord[1])
                    if intersect_anterior_wall:
                        break

                if intersect_anterior_wall:
                    print("This adhesion annotation is adjacent to the front abdominal wall")
                    annotations_to_front_wall += 1
                    # No need to check the rest of contour for this adhesion
                    annotations_to_contour += 1
                else:
                    # Check the whole contour
                    intersect_contour = False
                    for coord in contour_coord:
                        intersect_contour = adhesion.contains_point(coord[0], coord[1])
                        if intersect_contour:
                            break

                    if intersect_contour:
                        print("This adhesion annotation is adjacent to the abdominal cavity contour")
                        annotations_to_contour += 1


                adhesions_num += 1
        annotations_num += 1

    print("{} of {} annotated adhesions have information about contour".format(adhesions_num, annotated_num))
    print("{} of {} are adjacent to the front abdominal wall".format(annotations_to_front_wall, adhesions_num))
    print("{} of {} are adjacent to the abdominal cavity contour".format(annotations_to_contour, adhesions_num))


# A function to examine the detected front wall for all annotation for each
# abdominal cavity contour is available at the moment
def test_anterior_wall_detection(archive_path, visceral_slide_path):
    annotations_path = archive_path / METADATA_FOLDER / ANNOTATIONS_FILE_NAME
    inspexp_file_path = archive_path / METADATA_FOLDER / INSPEXP_FILE_NAME

    # load annotations
    annotations = load_bounding_boxes(annotations_path)
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
                exp_frame = scan_data[annotation.slice_id][1]
            except:
                print("Missing insp/exp data for the patient {}, scan {}, slice {}".format(annotation.patient_id,
                                                                                           annotation.scan_id,
                                                                                           annotation.slice_id))

            # Load the expiration frame (visceral slide is computed for the expiration frame)
            expiration_frame_path = archive_path / IMAGES_FOLDER / annotation.patient_id / annotation.scan_id / (
                    annotation.slice_id + ".mha")
            exp_frame = sitk.GetArrayFromImage(sitk.ReadImage(str(expiration_frame_path)))[exp_frame]

            verify_anterior_wall(x, y, exp_frame, annotation.scan_id)


# Allows to visually evaluate quality of the detection of front abdominal wall
# By plotting abdominal cavity contour over the frame and the detected front wall over the frame
# next to each other
def verify_anterior_wall(x, y, exp_frame, slice_id):
    out_path = Path("anterior_wall_results")
    out_path.mkdir(exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.imshow(exp_frame, cmap="gray")
    plt.axis('off')
    ax.scatter(x, y, s=4, color="r")

    anterior_wall_coord = get_anterior_wall_coord(x, y)

    ax = fig.add_subplot(122)
    plt.imshow(exp_frame, cmap="gray")
    plt.axis('off')
    ax.scatter(anterior_wall_coord[:, 0], anterior_wall_coord[:, 1], s=4, color="r")
    plt.axis('off')
    plt.savefig(out_path / (slice_id + ".png"), bbox_inches='tight', pad_inches=0)
    plt.close()


def get_anterior_wall_coord(x, y, connectivity_threshold=5):
    coords = np.column_stack((x, y))

    # Get unique y coordinates
    y_unique = np.unique(y)
    # For each unique y find the leftmost x
    x_left = []
    for y_current in y_unique:
        xs = [coord[0] for coord in coords if coord[1] == y_current]
        x_left.append(sorted(xs)[0])

    # Unique y coordinates and the corresponding leftmost x coordinates
    # Should give good approximation of the anterior abdominal wall
    x_left = np.array(x_left)
    anterior_wall_coord = np.column_stack((x_left, y_unique))

    # However, for some scans this approach fails because it will also keep pieces of
    # the countour on the right side (e.g. in pelvis area or due to poor quality segmentation)
    # Hence we first find all discontinuity points
    coord_prev = anterior_wall_coord[0]
    discontinuity_points = []
    for index in range(1, anterior_wall_coord.shape[0]):
        coord_cur = anterior_wall_coord[index]
        if abs(coord_cur[0] - coord_prev[0]) > connectivity_threshold:
            discontinuity_points.append(coord_prev)
        coord_prev = coord_cur

    # Then we remove small regions determined as a part of the front abdominal wall
    # The front wall is usually smooth enough not to have large gaps in x values,
    # hence this approach should be rather safe
    for discontinuity_point in discontinuity_points:
        index = np.where((anterior_wall_coord == discontinuity_point).all(axis=1))[0][0]
        before_len = index
        after_len = anterior_wall_coord.shape[0] - index
        if before_len < after_len:
            anterior_wall_coord = anterior_wall_coord[before_len + 1:]
        else:
            anterior_wall_coord = anterior_wall_coord[:before_len]
            # stop the loop since we removed all the next discontinuity points
            break

    return anterior_wall_coord


def test():
    archive_path = Path(ARCHIVE_PATH)
    visceral_slide_path = archive_path / "visceral_slide" / "visceral_slide"
    output_path = archive_path / "visceral_slide" / "annotated"

    #vis_annotation_and_vs(archive_path, visceral_slide_path, output_path)
    #test_anterior_wall_detection(archive_path, visceral_slide_path)
    annotations_to_abdominal_wall(archive_path, visceral_slide_path)

    """
    annotations = load_bounding_boxes(annotations_path)
    for annotation in annotations:
        print("Patient {}, scan {}, slice {}".format(annotation.patient_id, annotation.scan_id, annotation.slice_id))
        for adhesion in annotation.adhesions:
            print("Adhesion box x: {}, y: {}, width: {}, height: {}".format(adhesion.origin_x, adhesion.origin_y, adhesion.width, adhesion.height))

    """
    
    # Show randomly sampled annotation
    #annotation = random.choice(annotations)
    #show_annotation(annotation, archive_path)
    #save_annotated_gifs(annotations, archive_path)
    #annotate_mha(annotations, archive_path)

    

if __name__ == '__main__':
    test()
