"""Run nnu-net inference on all cases in datamodule"""
from cinemri.datamodules import CineMRIDataModule
from cinemri.config import ARCHIVE_PATH
from cinemri.definitions import CineMRISlice
from src.datasets import dev_dataset
from src.segmentation import run_full_inference
from src.vs_computation import (
    VSNormType,
    VSNormField,
    CumulativeVisceralSlideDetectorReg,
    CumulativeVisceralSlideDetectorDF,
)
from src.utils import load_visceral_slides
from src.detection_pipeline import (
    bb_with_threshold,
    predict_consecutive_minima,
    evaluate,
)
from src.adhesions import Adhesion, load_annotations
from pathlib import Path
import shutil
import SimpleITK as sitk
import numpy as np
import pickle
import json
import os
import datetime
import gc
import matplotlib.pyplot as plt


def get_dataset_with_boxes():
    datamodule = CineMRIDataModule(0, 0)
    datamodule.setup()

    return datamodule.train_dataset.dataset


def copy_dataset_to_dir(dataset, dest_dir):
    """Copy all slices in dataset to dir for nnunet inference"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    for id, filepath in dataset.images.items():
        if counter > 1000:
            break
        destination = (
            dest_dir / filepath.parts[-3] / filepath.parts[-2] / filepath.parts[-1]
        )
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(filepath, destination)
        counter += 1


def load_predictions(predictions_path):
    with open(predictions_path, "r") as file:
        predictions_dict = json.load(file)

    annotations = {}
    for patient_id, studies_dict in predictions_dict.items():
        for study_id, slices_dict in studies_dict.items():
            for slice_id, bounding_box_annotations in slices_dict.items():
                slice = CineMRISlice(slice_id, patient_id, study_id)
                bounding_boxes = []
                for bounding_box_annotation in bounding_box_annotations:
                    adhesion = Adhesion(bounding_box_annotation[0])
                    bounding_boxes.append((adhesion, bounding_box_annotation[1]))

                annotations[slice.full_id] = bounding_boxes

    return annotations


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


if __name__ == "__main__":
    # dataset = get_dataset_with_boxes()
    dataset = dev_dataset()

    # Output paths
    segmentation_result_dir = Path("/home/bram/data/registration_method/segmentations")
    visceral_slide_dir = Path("/home/bram/data/registration_method/visceral_slide")
    predictions_path = Path("/home/bram/data/registration_method/predictions.json")
    annotations_path = (
        ARCHIVE_PATH / "metadata" / "bounding_box_annotations_first_frame.json"
    )
    segmentation_result_dir.mkdir(exist_ok=True, parents=True)
    visceral_slide_dir.mkdir(exist_ok=True, parents=True)

    # Nnunet inference
    nnunet_input_dir = Path("/tmp/nnunet_input")
    nnunet_input_dir.mkdir(exist_ok=True, parents=True)
    nnunet_model_dir = Path(
        "/home/bram/repos/abdomenmrus-cinemri-vs-algorithm/nnunet/results"
    )
    if False:
        copy_dataset_to_dir(dataset, nnunet_input_dir)
        run_full_inference(
            nnunet_input_dir,
            segmentation_result_dir,
            nnunet_model_dir,
            nnUNet_task="Task101_AbdomenSegmentation",
        )

    # Registration + visceral slide computation
    if False:
        detector = CumulativeVisceralSlideDetectorReg()
        for idx, sample in enumerate(dataset):
            input_image_np = sample["numpy"]
            mask_path = (
                segmentation_result_dir
                / "merged_masks"
                / sample["PatientID"]
                / sample["StudyInstanceUID"]
                / (sample["SeriesInstanceUID"] + ".mha")
            )
            mask_sitk = sitk.ReadImage(str(mask_path))
            mask_np = sitk.GetArrayFromImage(mask_sitk)

            # Save visceral slide input
            vs_computation_input_path = (
                visceral_slide_dir
                / sample["PatientID"]
                / sample["StudyInstanceUID"]
                / sample["SeriesInstanceUID"]
                / "vs_computation_input.pkl"
            )
            vs_computation_input_path.parent.mkdir(exist_ok=True, parents=True)
            if vs_computation_input_path.is_file():
                print(f"Skipping {vs_computation_input_path}")
                print(f"{(idx+1)/len(dataset)}")
                continue
            x, y, values = detector.get_visceral_slide(
                input_image_np.astype(np.float32),
                mask_np,
                normalization_type=VSNormType.average_anterior_wall,
                normalization_field=VSNormField.complete,
                vs_computation_input_path=vs_computation_input_path,
            )

            # Save visceral slide
            pickle_path = (
                visceral_slide_dir
                / sample["PatientID"]
                / sample["StudyInstanceUID"]
                / sample["SeriesInstanceUID"]
                / "visceral_slide.pkl"
            )
            pickle_path.parent.mkdir(exist_ok=True, parents=True)
            slide_dict = {"x": x, "y": y, "slide": values}
            with open(pickle_path, "w+b") as file:
                pickle.dump(slide_dict, file)

            gc.collect()

    # Separate VS calculation
    if False:
        detector = CumulativeVisceralSlideDetectorDF()
        for sample in dataset:
            # Check pickling process
            input_image_np = sample["numpy"]

            # Load vs input
            vs_computation_input_path = (
                visceral_slide_dir
                / sample["PatientID"]
                / sample["StudyInstanceUID"]
                / sample["SeriesInstanceUID"]
                / "vs_computation_input.pkl"
            )
            with open(vs_computation_input_path, "r+b") as pkl_file:
                vs_computation_input = pickle.load(pkl_file)

            # Compute slide with separate method
            x, y, values = detector.get_visceral_slide(
                **vs_computation_input,
                normalization_type=VSNormType.average_anterior_wall,
            )

    # Detection
    if True:
        visceral_slides = load_visceral_slides(visceral_slide_dir)
        predictions = {}
        for visceral_slide in visceral_slides:
            patient_id, study_id, series_id = visceral_slide.full_id.split("_")
            prediction = bb_with_threshold(
                visceral_slide,
                (15, 15),
                (30, 30),
                (0, np.inf),
                pred_func=predict_consecutive_minima,
                apply_contour_prior=True,
                apply_curvature_filter=True,
            )
            prediction = [
                ([p.origin_x, p.origin_y, p.width, p.height], float(conf))
                for p, conf in prediction
            ]

            if patient_id not in predictions:
                predictions[patient_id] = {}
            if study_id not in predictions[patient_id]:
                predictions[patient_id][study_id] = {}
            predictions[patient_id][study_id][series_id] = prediction

        with open(predictions_path, "w") as file:
            json.dump(predictions, file)

    # Metrics
    if True:
        # Load predictions
        predictions = load_predictions(predictions_path)

        # Load annotations
        annotations = load_annotations(annotations_path, as_dict=True)

        evaluate(predictions, annotations, Path("/tmp"))

        # Make FROC
        # Make ROC
