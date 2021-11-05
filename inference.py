"""Run nnu-net inference on all cases in datamodule"""
from cinemri.datamodules import CineMRIDataModule
from src.segmentation import run_full_inference
from src.vs_computation import (
    VSNormType,
    VSNormField,
    CumulativeVisceralSlideDetectorReg,
)
from src.utils import load_visceral_slides
from src.detection_pipeline import bb_with_threshold
from pathlib import Path
import shutil
import SimpleITK as sitk
import numpy as np
import pickle
import json


def get_dataset_with_boxes():
    datamodule = CineMRIDataModule(0, 0)
    datamodule.setup()

    return datamodule.train_dataset.dataset


def copy_dataset_to_dir(dataset, dest_dir):
    """Copy all slices in dataset to dir for nnunet inference"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for id, filepath in dataset.images.items():
        destination = (
            dest_dir / filepath.parts[-3] / filepath.parts[-2] / filepath.parts[-1]
        )
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(filepath, destination)
        break


if __name__ == "__main__":
    dataset = get_dataset_with_boxes()

    # Output paths
    segmentation_result_dir = Path("/home/bram/data/registration_method/segmentations")
    visceral_slide_dir = Path("/home/bram/data/registration_method/visceral_slide")
    predictions_path = Path("/home/bram/data/registration_method/predictions.json")
    segmentation_result_dir.mkdir(exist_ok=True, parents=True)
    visceral_slide_dir.mkdir(exist_ok=True, parents=True)

    # Nnunet inference
    nnunet_input_dir = Path("/tmp/nnunet_input")
    nnunet_input_dir.mkdir(exist_ok=True, parents=True)
    copy_dataset_to_dir(dataset, nnunet_input_dir)
    nnunet_model_dir = Path(
        "/home/bram/repos/abdomenmrus-cinemri-vs-algorithm/nnunet/results"
    )
    if False:
        run_full_inference(
            nnunet_input_dir,
            segmentation_result_dir,
            nnunet_model_dir,
            nnUNet_task="Task101_AbdomenSegmentation",
        )

    # Registration + visceral slide computation
    if False:
        detector = CumulativeVisceralSlideDetectorReg()
        for sample in dataset:
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
            x, y, values = detector.get_visceral_slide(
                input_image_np.astype(np.float32),
                mask_np,
                normalization_type=VSNormType.average_anterior_wall,
                normalization_field=VSNormField.complete,
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

    # Detection
    if True:
        visceral_slides = load_visceral_slides(visceral_slide_dir)
        predictions = {}
        for visceral_slide in visceral_slides:
            patient_id, study_id, series_id = visceral_slide.full_id.split("_")
            prediction = bb_with_threshold(visceral_slide, (15, 15), (30, 30), (0, 100))
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
