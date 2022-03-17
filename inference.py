"""Run nnu-net inference on all cases in datamodule"""
from cinemri.datamodules import CineMRIDataModule
from cinemri.config import ARCHIVE_PATH
from cinemri.definitions import CineMRISlice
from cinemri.visualisation import plot_frame
from src.datasets import dev_dataset
from src.segmentation import run_full_inference
from src.classification import get_boxes_from_raw
from src.vs_computation import (
    VSNormType,
    VSNormField,
    CumulativeVisceralSlideDetectorReg,
    CumulativeVisceralSlideDetectorDF,
    FirstToAllVisceralSlideDetectorReg,
    FirstToAllVisceralSlideDetectorDF,
    calculate_motion_map,
)
from src.utils import load_visceral_slides
from src.detection_pipeline import (
    bb_with_threshold,
    predict_consecutive_minima,
    evaluate,
)
from src.adhesions import AdhesionType, Adhesion, load_annotations, load_predictions
from src.contour import (
    get_adhesions_prior_coords,
    Evaluation,
    filter_out_prior_vs_subset,
)
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
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from report_guided_annotation.extract_lesion_candidates import preprocess_softmax


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
        # detector = FirstToAllVisceralSlideDetectorReg()
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
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
            # if vs_computation_input_path.is_file():
            #     print(f"Skipping {vs_computation_input_path}")
            #     print(f"{(idx+1)/len(dataset)}")
            #     continue
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

    # Separate VS calculation
    if False:
        visceral_slide_dir = Path(
            "/home/bram/data/registration_method/visceral_slide_first_to_all"
        )
        visceral_slide_dir_recompute = Path(
            "/home/bram/data/registration_method/visceral_slide_first_to_all_mean"
        )
        detector = CumulativeVisceralSlideDetectorDF()
        detector = FirstToAllVisceralSlideDetectorDF()
        for sample in tqdm(dataset, desc="Calculating visceral slide"):
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

            if "warping_dfs" not in vs_computation_input:
                vs_computation_input["warping_dfs"] = []
            # Compute slide with separate method
            x, y, values = detector.get_visceral_slide(
                **vs_computation_input,
                normalization_type=VSNormType.none,
            )

            # Save visceral slide
            pickle_path = (
                visceral_slide_dir_recompute
                / sample["PatientID"]
                / sample["StudyInstanceUID"]
                / sample["SeriesInstanceUID"]
                / "visceral_slide.pkl"
            )
            pickle_path.parent.mkdir(exist_ok=True, parents=True)
            slide_dict = {"x": x, "y": y, "slide": values}
            with open(pickle_path, "w+b") as file:
                pickle.dump(slide_dict, file)

    # Generate pixel dataset
    if True:
        # For all series
        # For all pixels in contour
        #
        # Determine label by overlap with box
        # Label be multiclass:
        # 0: background
        # 1: anterior wall
        # 2: pelvis
        # 3: interior
        #
        # Save the following features to disk
        #
        # first to all mean visceral slide
        # anterior/top/pelvis/posterior
        # x/y position in index coordinates
        # clockwise portion of contour part (0-1)
        #
        # optional for future:
        #
        # curvature
        # contrast difference left and right of contour
        # motion values, e.g. mean motion map, local motion estimate
        # min and max visceral slide over all registrations
        #
        visceral_slide_dir_recompute = Path(
            "/home/bram/data/registration_method/visceral_slide_first_to_all_mean"
        )
        visceral_slides = load_visceral_slides(visceral_slide_dir_recompute)
        annotations = load_predictions(extended_annotations_path)

        features = {}
        series_ids = []
        patient_ids = []
        for visceral_slide in tqdm(
            visceral_slides, desc="Assembling classifier features"
        ):
            patient_id, study_id, series_id = visceral_slide.full_id.split("_")
            series_ids.append(series_id)
            patient_ids.append(patient_id)
            sample = dataset[series_id]
            annotation = annotations[visceral_slide.full_id]

            # Determine label
            x_label = visceral_slide.x
            y_label = visceral_slide.y
            label = np.zeros_like(visceral_slide.x)
            for adhesion, _ in annotation:
                for idx, (x, y) in enumerate(zip(x_label, y_label)):
                    if adhesion.contains_point(x, y):
                        if adhesion.type == AdhesionType.anteriorWall:
                            label[idx] = 1
                        if adhesion.type == AdhesionType.pelvis:
                            label[idx] = 2
                        if adhesion.type == AdhesionType.inside:
                            label[idx] = 3

            # Aggregate features
            features[series_id] = {}
            features[series_id]["slide"] = list(visceral_slide.values)
            features[series_id]["x"] = list(visceral_slide.x)
            features[series_id]["y"] = list(visceral_slide.y)
            features[series_id]["label"] = list(label > 0)

        # TODO make one long list of all cases

        # TODO save features to disk

        # Plot sample with box and label
        # boxes = []
        # for box in sample["box"]:
        #     boxes.append({"box": box, "color": "green"})
        # fig, ax = plt.subplots()
        # plot_frame(ax, sample["numpy"][0], boxes=boxes)
        # vs_scatter = ax.scatter(
        #     x_label, y_label, vmin=0, vmax=3, s=5, c=label, cmap="jet"
        # )
        # plt.show()

    # Train and val classifier
    if False:
        evaluation = Evaluation.anterior_wall
        # evaluation = Evaluation.pelvis

        def assemble_features(features, series_ids):
            """Go from dict series_id->feature->list to sklearn format
            vectors"""
            included_features = ["slide", "x", "y"]
            assembled = {}
            label = []
            mask = []
            for feature_label in included_features:
                assembled[feature_label] = []

            for series_id in series_ids:
                for feature_label in included_features:
                    assembled[feature_label] += features[series_id][feature_label]
                label += features[series_id]["label"]

                # Get contour part masks
                x = features[series_id]["x"]
                y = features[series_id]["y"]
                case_mask = get_adhesions_prior_coords(
                    x, y, evaluation=evaluation, return_mask=True
                )
                mask += list(case_mask)

            feature_array = np.zeros((len(label), len(included_features)))
            for idx, feature_label in enumerate(included_features):
                feature_array[:, idx] = assembled[feature_label]

            return feature_array[mask], np.array(label)[mask]

        def get_normalizer(train_features):
            return StandardScaler().fit(train_features)

        cv = GroupKFold()
        predictions_dict = {}
        for train_index, test_index in cv.split(series_ids, groups=patient_ids):
            train_series, test_series = (
                np.array(series_ids)[train_index],
                np.array(series_ids)[test_index],
            )
            train_features, train_labels = assemble_features(features, train_series)
            test_features, test_labels = assemble_features(features, test_series)
            normalizer = get_normalizer(train_features)

            # Fit Logistic regression classifier
            # clf = LogisticRegression().fit(train_features, train_labels)
            print("Training classifier")
            clf = SVC(class_weight="balanced", probability=True, max_iter=-1).fit(
                normalizer.transform(train_features),
                train_labels,
            )
            print("Done!")
            # print("Train acc")
            # print(clf.score(train_features, train_labels))
            # print("Val acc")
            # print(clf.score(test_features, test_labels))

            # Predict all pixels on validation set
            for series_id in test_series:
                test_features_unnorm, test_labels = assemble_features(
                    features, [series_id]
                )
                test_features = normalizer.transform(test_features_unnorm)
                prediction = clf.predict_proba(test_features)[:, 1]

                # Convert to connected predictions
                # TODO get x,y somehow nicer from features
                pred_boxes = get_boxes_from_raw(
                    prediction, test_features_unnorm[:, 1], test_features_unnorm[:, 2]
                )

                # Get patient id and study id
                sample = dataset[str(series_id)]
                patient_id = sample["PatientID"]
                study_id = sample["StudyInstanceUID"]

                # Save in predictions_dict
                prediction_list = []
                for p, conf in pred_boxes:
                    box = [p.origin_x, p.origin_y, p.width, p.height]
                    conf = float(conf)

                    # p.assign_type_from_mask(mask_np)
                    # if p.type == AdhesionType.unset:
                    #     box_type = "unset"
                    # if p.type == AdhesionType.pelvis:
                    #     box_type = "pelvis"
                    # if p.type == AdhesionType.anteriorWall:
                    #     box_type = "anterior"
                    # if p.type == AdhesionType.inside:
                    #     box_type = "inside"
                    #
                    box_type = "anterior"
                    prediction_list.append((box, conf, box_type))
                if patient_id not in predictions_dict:
                    predictions_dict[patient_id] = {}
                if study_id not in predictions_dict[patient_id]:
                    predictions_dict[patient_id][study_id] = {}
                predictions_dict[patient_id][study_id][str(series_id)] = prediction_list

                # Plot predictions and determine how to make boxes from them
                # sample = dataset[str(series_id)]
                # boxes = []
                # for box in sample["box"]:
                #     boxes.append({"box": box, "color": "green"})
                # for p, conf in pred_boxes:
                #     color = "red"
                #     box = [p.origin_x, p.origin_y, p.width, p.height]
                #     boxes.append({"box": box, "color": color, "label": f"{conf:.2f}"})

                # fig = plt.figure()
                # ax_raw = plt.subplot(121)
                # plot_frame(ax_raw, sample["numpy"][0], boxes=boxes)
                # x, y = get_adhesions_prior_coords(
                #     features[series_id]["x"],
                #     features[series_id]["y"],
                #     evaluation=evaluation,
                # )
                # vs_scatter = ax_raw.scatter(
                #     x,
                #     y,
                #     vmin=0,
                #     vmax=1,
                #     s=5,
                #     c=prediction,
                #     cmap="jet",
                # )
                # plt.title(np.max(prediction))
                # ax_processed = plt.subplot(122)
                # plot_frame(ax_processed, sample["numpy"][0], boxes=boxes)
                # x, y = get_adhesions_prior_coords(
                #     features[series_id]["x"],
                #     features[series_id]["y"],
                #     evaluation=evaluation,
                # )
                # vs_scatter = ax_processed.scatter(
                #     x,
                #     y,
                #     s=5,
                #     c=prediction,
                #     cmap="jet",
                # )
                # plt.title(np.max(prediction))
                # plt.show()
        with open(predictions_path, "w") as file:
            json.dump(predictions_dict, file)

    # Write plots for pipeline visualization
    if True:
        image_dir = Path("/tmp/visceral_slide_pipeline")

        # Load image, segmentation, visceral slide
        series_id = series_ids[0]
        sample = dataset[series_id]
        mask_path = (
            segmentation_result_dir
            / "merged_masks"
            / sample["PatientID"]
            / sample["StudyInstanceUID"]
            / (sample["SeriesInstanceUID"] + ".mha")
        )
        mask_sitk = sitk.ReadImage(str(mask_path))
        mask_np = sitk.GetArrayFromImage(mask_sitk)
        sample_np = sample["numpy"]

        # Vanilla plot
        fig, ax = plt.subplots()
        plot_frame(ax, sample_np[0])
        fig.savefig(image_dir / "0-first-frame.png", bbox_inches="tight", pad_inches=0)

        # Plot with cavity segmentation
        fig, ax = plt.subplots()
        plot_frame(ax, sample_np[0])
        ax.imshow(mask_np[0], alpha=mask_np[0] * 0.5)
        fig.savefig(
            image_dir / "1-cavity-segmentation.png", bbox_inches="tight", pad_inches=0
        )

        # Cavity and surroundings plot
        fig, ax = plt.subplots()
        plot_frame(ax, sample_np[0] * mask_np[0])
        fig.savefig(image_dir / "2-cavity.png", bbox_inches="tight", pad_inches=0)
        fig, ax = plt.subplots()
        plot_frame(ax, sample_np[0] * (1 - mask_np[0]))
        fig.savefig(image_dir / "2-rest.png", bbox_inches="tight", pad_inches=0)

        # Visceral slide plot
        vs = visceral_slides[0]
        vs = filter_out_prior_vs_subset(vs.x, vs.y, vs.values)
        fig, ax = plt.subplots()
        plot_frame(ax, sample_np[0])
        ax.scatter(
            vs[:, 0],
            vs[:, 1],
            s=5,
            c=vs[:, 2],
            cmap="jet",
        )
        fig.savefig(
            image_dir / "3-visceral_slide.png", bbox_inches="tight", pad_inches=0
        )

        # Plot with bounding boxes
        predictions = load_predictions(predictions_path)
        for full_id in predictions:
            _, _, series = full_id.split("_")
            if series == series_id:
                prediction = predictions[full_id]

        prediction_list = []
        for entry in prediction:
            p = entry[0]
            conf = entry[1]
            box = [p.origin_x, p.origin_y, p.width, p.height]
            conf = float(conf)
            type = p.type

            prediction_list.append((box, conf, type))
        boxes = []
        for box in prediction_list:
            color = "red"
            boxes.append({"box": box[0], "color": color, "label": ""})
        fig, ax = plt.subplots()
        plot_frame(ax, sample_np[0], boxes=boxes)
        fig.savefig(
            image_dir / "4-bounding-boxes.png", bbox_inches="tight", pad_inches=0
        )
        # plt.show()
    # Detection
    if False:
        visceral_slide_dir_recompute = Path(
            "/home/bram/data/registration_method/visceral_slide_first_to_all_mean"
        )
        visceral_slides = load_visceral_slides(visceral_slide_dir_recompute)
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
                (20, 20),
                (0, np.inf),
                pred_func=predict_consecutive_minima,
                apply_contour_prior=True,
                apply_curvature_filter=False,
            )

            # Assemble predictions for json format
            prediction_list = []
            for p, conf in prediction:
                box = [p.origin_x, p.origin_y, p.width, p.height]
                conf = float(conf)

                # p.assign_type_from_mask(mask_np)
                # if p.type == AdhesionType.unset:
                #     box_type = "unset"
                # if p.type == AdhesionType.pelvis:
                #     box_type = "pelvis"
                # if p.type == AdhesionType.anteriorWall:
                #     box_type = "anterior"
                # if p.type == AdhesionType.inside:
                #     box_type = "inside"
                box_type = "anterior"
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
    if False:
        predictions_path = predictions_path.parent / "predictions.json"
        # Load predictions
        predictions = load_predictions(predictions_path)

        # Load annotations
        annotations = load_predictions(extended_annotations_path)

        metrics = picai_eval(
            predictions, annotations, flat=True, types=[AdhesionType.inside]
        )

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
