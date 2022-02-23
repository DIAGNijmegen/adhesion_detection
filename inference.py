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
from src.adhesions import AdhesionType, Adhesion, load_annotations, load_predictions
from src.evaluation import picai_eval
from pathlib import Path
import shutil
import SimpleITK as sitk
import numpy as np
import pickle
import json
import os
import datetime
import gc
from tqdm import tqdm
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


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


if __name__ == "__main__":
    # dataset = get_dataset_with_boxes()
    dataset = dev_dataset()

    # Output paths
    segmentation_result_dir = Path("/home/bram/data/registration_method/segmentations")
    visceral_slide_dir = Path("/home/bram/data/registration_method/visceral_slide")
    predictions_path = Path(
        "/home/bram/data/registration_method/predictions/predictions.json"
    )
    extended_annotations_path = Path(
        "/home/bram/data/registration_method/extended_annotations.json"
    )
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
                normalization_type=VSNormType.none,
            )

    # Detection
    if False:
        visceral_slides = load_visceral_slides(visceral_slide_dir)
        predictions = {}
        for visceral_slide in tqdm(visceral_slides):
            patient_id, study_id, series_id = visceral_slide.full_id.split("_")

            # Load mask for detection of adhesion type
            mask_path = (
                segmentation_result_dir
                / "merged_masks"
                / patient_id
                / study_id
                / (series_id + ".mha")
            )
            mask_sitk = sitk.ReadImage(str(mask_path))
            mask_np = sitk.GetArrayFromImage(mask_sitk)

            # Get all predicted boxes
            prediction = bb_with_threshold(
                visceral_slide,
                (15, 15),
                (30, 30),
                (0, np.inf),
                pred_func=predict_consecutive_minima,
                apply_contour_prior=True,
                apply_curvature_filter=True,
            )

            # Assemble predictions for json format
            prediction_list = []
            for p, conf in prediction:
                box = [p.origin_x, p.origin_y, p.width, p.height]
                conf = float(conf)

                p.assign_type_from_mask(mask_np)
                if p.type == AdhesionType.unset:
                    box_type = "unset"
                if p.type == AdhesionType.pelvis:
                    box_type = "pelvis"
                if p.type == AdhesionType.anteriorWall:
                    box_type = "anterior"
                if p.type == AdhesionType.inside:
                    box_type = "inside"
                prediction_list.append((box, conf, box_type))

            if patient_id not in predictions:
                predictions[patient_id] = {}
            if study_id not in predictions[patient_id]:
                predictions[patient_id][study_id] = {}
            predictions[patient_id][study_id][series_id] = prediction_list

        with open(predictions_path, "w") as file:
            json.dump(predictions, file)

    # Write reference standard as json
    if False:
        visceral_slides = load_visceral_slides(visceral_slide_dir)
        predictions = {}
        for sample in tqdm(dataset):
            patient_id = sample["PatientID"]
            study_id = sample["StudyInstanceUID"]
            series_id = sample["SeriesInstanceUID"]

            # Load mask for detection of adhesion type
            mask_path = (
                segmentation_result_dir
                / "merged_masks"
                / patient_id
                / study_id
                / (series_id + ".mha")
            )
            mask_sitk = sitk.ReadImage(str(mask_path))
            mask_np = sitk.GetArrayFromImage(mask_sitk)

            # Get all reference boxes
            prediction = sample["box"]

            # Assemble for json format
            prediction_list = []
            for box in prediction:
                p = Adhesion(box)
                conf = 1

                p.assign_type_from_mask(mask_np)
                if p.type == AdhesionType.unset:
                    box_type = "unset"
                if p.type == AdhesionType.pelvis:
                    box_type = "pelvis"
                if p.type == AdhesionType.anteriorWall:
                    box_type = "anterior"
                if p.type == AdhesionType.inside:
                    box_type = "inside"
                prediction_list.append((box, conf, box_type))

            if patient_id not in predictions:
                predictions[patient_id] = {}
            if study_id not in predictions[patient_id]:
                predictions[patient_id][study_id] = {}
            predictions[patient_id][study_id][series_id] = prediction_list

        with open(extended_annotations_path, "w") as file:
            json.dump(predictions, file)

    # Metrics
    if True:
        # Load predictions
        predictions = load_predictions(predictions_path)

        # Load annotations
        annotations = load_predictions(extended_annotations_path)

        metrics = picai_eval(predictions, annotations, flat=True)

        # Plot FROC
        plt.figure()
        plt.xlabel("Mean number of FPs per image")
        plt.ylabel("Sensitivity")
        plt.ylim([0, 1])
        plt.xscale("log")
        plt.plot(metrics["FP_per_case"], metrics["sensitivity"])
        plt.show()

        # Plot ROC
        plt.figure()
        plt.plot(metrics["fpr"], metrics["tpr"])
        plt.show()
