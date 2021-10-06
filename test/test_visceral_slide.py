from unittest import TestCase
from pathlib import Path
import numpy as np
import pickle
import json
from visceral_slide import VisceralSlideDetectorReg, VisceralSlideDetectorDF, CumulativeVisceralSlideDetectorReg,\
    CumulativeVisceralSlideDetectorDF, VSNormType, VSNormField, VSWarpingField
import SimpleITK as sitk
import matplotlib.pyplot as plt

PATIENT_ID = "CM0020"
STUDY_ID = "1.2.752.24.7.621449243.4474616"
SLICE_ID = "1.3.12.2.1107.5.2.30.26380.2019060311131653190024186.0.0.0"
plot = False

class TestVisceralSlideDetector(TestCase):

    def test_get_full_deformation_field(self):
        exp_path = Path("data/exp.npy")
        exp_image = np.load(exp_path).astype(np.uint32)

        exp_mask_path = Path("data/exp_mask.npy")
        exp_mask_image = np.load(exp_mask_path).astype(np.uint32)

        insp_path = Path("data/insp.npy")
        insp_image = np.load(insp_path).astype(np.uint32)

        insp_mask_path = Path("data/insp_mask.npy")
        insp_mask_image = np.load(insp_mask_path).astype(np.uint32)

        masked_field = VisceralSlideDetectorReg().get_full_deformation_field(exp_image, insp_image, exp_mask_image, insp_mask_image)

        self.assertTrue(masked_field[..., 0][insp_mask_image == 0].any(),
                        "Full deformation filed is expeced, but it is masked with abdominal cavity")
        self.assertTrue(masked_field[..., 1][insp_mask_image == 0].any(),
                        "Full deformation filed is expeced, but it is masked with abdominal cavity")

        self.assertTrue(masked_field[..., 0][insp_mask_image == 1].any(),
                        "Full deformation filed is expeced, but it is masked with abdominal cavity surroundings")
        self.assertTrue(masked_field[..., 1][insp_mask_image == 1].any(),
                        "Full deformation filed is expeced, but it is masked with abdominal cavity surroundings")

    def test__calculate_visceral_slide(self):
        exp_path = Path("data/exp.npy")
        exp_mask_path = Path("data/exp_mask.npy")
        insp_path = Path("data/insp.npy")
        insp_mask_path = Path("data/insp_mask.npy")
        expected_visceral_slide_fixed_mask_path = Path("data/expected_slide_with_fixed_mask.pkl")
        expected_visceral_slide_no_fixed_mask_path = Path("data/expected_slide_without_fixed_mask.pkl")

        exp_image = np.load(exp_path).astype(np.uint32)
        exp_mask_image = np.load(exp_mask_path).astype(np.uint8)
        insp_image = np.load(insp_path).astype(np.uint32)
        insp_mask_image = np.load(insp_mask_path).astype(np.uint8)

        detector = VisceralSlideDetectorReg()
        x, y, slide = detector.get_visceral_slide(exp_image, exp_mask_image, insp_image, insp_mask_image)
        visceral_slide_fixed_mask = {"x": x, "y": y, "slide": slide}

        with open(expected_visceral_slide_fixed_mask_path, "r+b") as file:
            expected_visceral_slide_fixed_mask = pickle.load(file)
            
        self.assertTrue(np.array_equal(visceral_slide_fixed_mask["x"], expected_visceral_slide_fixed_mask["x"]),
                        "Incorrect x coordinates of contour computed with fixed mask")
        self.assertTrue(np.array_equal(visceral_slide_fixed_mask["y"], expected_visceral_slide_fixed_mask["y"]),
                        "Incorrect y coordinates of contour computed with fixed mask")
        self.assertTrue(np.array_equal(visceral_slide_fixed_mask["slide"], expected_visceral_slide_fixed_mask["slide"]),
                        "Incorrect visceral slide computed with fixed mask")

        x, y, slide = detector.get_visceral_slide(insp_image, insp_mask_image, exp_image)
        visceral_slide_no_fixed_mask = {"x": x, "y": y, "slide": slide}

        with open(expected_visceral_slide_no_fixed_mask_path, "r+b") as file:
            expected_visceral_slide_no_fixed_mask = pickle.load(file)

        self.assertTrue(np.array_equal(visceral_slide_no_fixed_mask["x"], expected_visceral_slide_no_fixed_mask["x"]),
                        "Incorrect x coordinates of contour computed without fixed mask")
        self.assertTrue(np.array_equal(visceral_slide_no_fixed_mask["y"], expected_visceral_slide_no_fixed_mask["y"]),
                        "Incorrect y coordinates of contour computed without fixed mask")
        self.assertTrue(np.array_equal(visceral_slide_no_fixed_mask["slide"], expected_visceral_slide_no_fixed_mask["slide"]),
                        "Incorrect visceral slide computed without fixed mask")

    def test_insp_exp_no_norm(self):

        with open("vs_load_check_insp_exp/expected_vs_insp_exp_no_norm.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        with open("vs_load_check_insp_exp/inspexp.json") as inspexp_file:
            inspexp_data = json.load(inspexp_file)

        inspexp_frames = inspexp_data[PATIENT_ID][STUDY_ID][SLICE_ID]
        insp_ind, exp_ind = inspexp_frames[0], inspexp_frames[1]

        slice_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/slice.mha"))
        insp_frame, exp_frame = slice_array[insp_ind], slice_array[exp_ind]
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/mask.mha"))
        insp_mask, exp_mask = mask_array[insp_ind], mask_array[exp_ind]

        # With image registration performed on the fly
        detector1 = VisceralSlideDetectorReg()
        x1, y1, vs1 = detector1.get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask)

        if plot:
            plt.figure()
            plt.imshow(insp_frame, cmap="gray")
            plt.scatter(x1, y1, s=5, c=vs1, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_df_calc_insp_exp.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        self.assertTrue(np.array_equal(expected_x, x1),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y1),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs1),
                        "Incorrect visceral slide")

        # With DFs loading
        df_cavity = np.load("vs_load_check_insp_exp/df_cavity.npy")
        df_rest = np.load("vs_load_check_insp_exp/df_rest.npy")
        df_complete = np.load("vs_load_check_insp_exp/df_complete.npy")
        moving_mask = np.load("vs_load_check_insp_exp/moving_mask.npy")

        detector2 = VisceralSlideDetectorDF()
        x2, y2, vs2 = detector2.get_visceral_slide(df_cavity, df_rest, df_complete, moving_mask)

        self.assertTrue(np.array_equal(expected_x, x2),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y2),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs2),
                        "Incorrect visceral slide")

    def test_insp_exp_norm_avg_anterior_rest(self):

        with open("vs_load_check_insp_exp/expected_vs_insp_exp_norm_avg_anterior_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        with open("vs_load_check_insp_exp/inspexp.json") as inspexp_file:
            inspexp_data = json.load(inspexp_file)

        inspexp_frames = inspexp_data[PATIENT_ID][STUDY_ID][SLICE_ID]
        insp_ind, exp_ind = inspexp_frames[0], inspexp_frames[1]

        slice_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/slice.mha"))
        insp_frame, exp_frame = slice_array[insp_ind], slice_array[exp_ind]
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/mask.mha"))
        insp_mask, exp_mask = mask_array[insp_ind], mask_array[exp_ind]

        # With image registration performed on the fly
        detector1 = VisceralSlideDetectorReg()
        x1, y1, vs1 = detector1.get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask,
                                                   VSNormType.average_anterior_wall, VSNormField.rest)

        if plot:
            plt.figure()
            plt.imshow(insp_frame, cmap="gray")
            plt.scatter(x1, y1, s=5, c=vs1, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_df_calc_insp_exp_norm_avg_anterior_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        self.assertTrue(np.array_equal(expected_x, x1),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y1),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs1),
                        "Incorrect visceral slide")

        # With DFs loading
        df_cavity = np.load("vs_load_check_insp_exp/df_cavity.npy")
        df_rest = np.load("vs_load_check_insp_exp/df_rest.npy")
        moving_mask = np.load("vs_load_check_insp_exp/moving_mask.npy")

        detector2 = VisceralSlideDetectorDF()
        x2, y2, vs2 = detector2.get_visceral_slide(df_cavity, df_rest, df_rest, moving_mask,
                                                   VSNormType.average_anterior_wall)

        self.assertTrue(np.array_equal(expected_x, x2),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y2),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs2),
                        "Incorrect visceral slide")

    def test_insp_exp_norm_avg_anterior_complete(self):

        with open("vs_load_check_insp_exp/expected_vs_insp_exp_norm_avg_anterior_complete.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        with open("vs_load_check_insp_exp/inspexp.json") as inspexp_file:
            inspexp_data = json.load(inspexp_file)

        inspexp_frames = inspexp_data[PATIENT_ID][STUDY_ID][SLICE_ID]
        insp_ind, exp_ind = inspexp_frames[0], inspexp_frames[1]

        slice_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/slice.mha"))
        insp_frame, exp_frame = slice_array[insp_ind], slice_array[exp_ind]
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/mask.mha"))
        insp_mask, exp_mask = mask_array[insp_ind], mask_array[exp_ind]

        # With image registration performed on the fly
        detector1 = VisceralSlideDetectorReg()
        x1, y1, vs1 = detector1.get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask,
                                                   VSNormType.average_anterior_wall, VSNormField.complete)

        if plot:
            plt.figure()
            plt.imshow(insp_frame, cmap="gray")
            plt.scatter(x1, y1, s=5, c=vs1, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_df_calc_insp_exp_norm_avg_anterior_complete.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        self.assertTrue(np.array_equal(expected_x, x1),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y1),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs1),
                        "Incorrect visceral slide")

        # With DFs loading
        df_cavity = np.load("vs_load_check_insp_exp/df_cavity.npy")
        df_rest = np.load("vs_load_check_insp_exp/df_rest.npy")
        df_complete = np.load("vs_load_check_insp_exp/df_complete.npy")
        moving_mask = np.load("vs_load_check_insp_exp/moving_mask.npy")

        detector2 = VisceralSlideDetectorDF()
        x2, y2, vs2 = detector2.get_visceral_slide(df_cavity, df_rest, df_complete, moving_mask,
                                                   VSNormType.average_anterior_wall)

        self.assertTrue(np.array_equal(expected_x, x2),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y2),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs2),
                        "Incorrect visceral slide")

    def test_insp_exp_norm_vicinity_rest(self):

        with open("vs_load_check_insp_exp/expected_vs_insp_exp_norm_vicinity_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        with open("vs_load_check_insp_exp/inspexp.json") as inspexp_file:
            inspexp_data = json.load(inspexp_file)

        inspexp_frames = inspexp_data[PATIENT_ID][STUDY_ID][SLICE_ID]
        insp_ind, exp_ind = inspexp_frames[0], inspexp_frames[1]

        slice_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/slice.mha"))
        insp_frame, exp_frame = slice_array[insp_ind], slice_array[exp_ind]
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/mask.mha"))
        insp_mask, exp_mask = mask_array[insp_ind], mask_array[exp_ind]

        # With image registration performed on the fly
        detector1 = VisceralSlideDetectorReg()
        x1, y1, vs1 = detector1.get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask,
                                                   VSNormType.contour_vicinity, VSNormField.rest)

        if plot:
            plt.figure()
            plt.imshow(insp_frame, cmap="gray")
            plt.scatter(x1, y1, s=5, c=vs1, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_df_calc_insp_exp_norm_vicinity_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        self.assertTrue(np.array_equal(expected_x, x1),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y1),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs1),
                        "Incorrect visceral slide")

        # With DFs loading
        df_cavity = np.load("vs_load_check_insp_exp/df_cavity.npy")
        df_rest = np.load("vs_load_check_insp_exp/df_rest.npy")
        moving_mask = np.load("vs_load_check_insp_exp/moving_mask.npy")

        detector2 = VisceralSlideDetectorDF()
        x2, y2, vs2 = detector2.get_visceral_slide(df_cavity, df_rest, df_rest, moving_mask,
                                                   VSNormType.contour_vicinity)

        self.assertTrue(np.array_equal(expected_x, x2),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y2),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs2),
                        "Incorrect visceral slide")

    def test_insp_exp_norm_vicinity_complete(self):

        with open("vs_load_check_insp_exp/expected_vs_insp_exp_norm_vicinity_complete.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        with open("vs_load_check_insp_exp/inspexp.json") as inspexp_file:
            inspexp_data = json.load(inspexp_file)

        inspexp_frames = inspexp_data[PATIENT_ID][STUDY_ID][SLICE_ID]
        insp_ind, exp_ind = inspexp_frames[0], inspexp_frames[1]

        slice_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/slice.mha"))
        insp_frame, exp_frame = slice_array[insp_ind], slice_array[exp_ind]
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_insp_exp/mask.mha"))
        insp_mask, exp_mask = mask_array[insp_ind], mask_array[exp_ind]

        # With image registration performed on the fly
        detector1 = VisceralSlideDetectorReg()
        x1, y1, vs1 = detector1.get_visceral_slide(insp_frame, insp_mask, exp_frame, exp_mask,
                                                   VSNormType.contour_vicinity, VSNormField.complete)

        if plot:
            plt.figure()
            plt.imshow(insp_frame, cmap="gray")
            plt.scatter(x1, y1, s=5, c=vs1, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_df_calc_insp_exp_norm_vicinity_complete.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        self.assertTrue(np.array_equal(expected_x, x1),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y1),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs1),
                        "Incorrect visceral slide")

        # With DFs loading
        df_cavity = np.load("vs_load_check_insp_exp/df_cavity.npy")
        df_rest = np.load("vs_load_check_insp_exp/df_rest.npy")
        df_complete = np.load("vs_load_check_insp_exp/df_complete.npy")
        moving_mask = np.load("vs_load_check_insp_exp/moving_mask.npy")

        detector2 = VisceralSlideDetectorDF()
        x2, y2, vs2 = detector2.get_visceral_slide(df_cavity, df_rest, df_complete, moving_mask,
                                                   VSNormType.contour_vicinity)

        self.assertTrue(np.array_equal(expected_x, x2),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y2),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs2),
                        "Incorrect visceral slide")

    def load_sequences(self, path, pattern="[0-9]*.npy"):
        files_glob = path.glob(pattern)
        files = [file.name for file in files_glob]
        files = sorted([file for file in files], key=lambda file_id: int(file_id[:-4].split("_")[-1]))
        return [np.load(path / file) for file in files]

    def test_cum_vs_no_norm_warp_contour(self):

        with open("vs_load_check_cum/expected_cum_no_norm_warp_contour.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        contour_dfs = self.load_sequences(Path("vs_load_check_cum/df_contour"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               contour_dfs,
                                               rest_dfs,
                                               VSNormType.none)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_no_norm_warp_contour.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()

        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.contours,
                                               VSNormType.none,
                                               VSNormField.rest)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")


    def test_cum_vs_no_norm_warp_rest(self):

        with open("vs_load_check_cum/expected_cum_no_norm_warp_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               VSNormType.none)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_no_norm_warp_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()

        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.rest,
                                               VSNormType.none,
                                               VSNormField.rest)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_avg_norm_rest_warp_rest(self):

        with open("vs_load_check_cum/expected_cum_norm_avg_norm_rest_warp_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               VSNormType.average_anterior_wall)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_avg_norm_rest_warp_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()

        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.rest,
                                               VSNormType.average_anterior_wall,
                                               VSNormField.rest)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_avg_norm_rest_warp_contour(self):

        with open("vs_load_check_cum/expected_cum_norm_avg_norm_rest_warp_contour.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        contour_dfs = self.load_sequences(Path("vs_load_check_cum/df_contour"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               contour_dfs,
                                               rest_dfs,
                                               VSNormType.average_anterior_wall)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_avg_norm_rest_warp_contour.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.contours,
                                               VSNormType.average_anterior_wall,
                                               VSNormField.rest)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_avg_norm_complete_warp_rest(self):

        with open("vs_load_check_cum/expected_cum_norm_avg_norm_complete_warp_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        complete_dfs = self.load_sequences(Path("vs_load_check_cum/df_complete"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               complete_dfs,
                                               VSNormType.average_anterior_wall)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_avg_norm_complete_warp_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.rest,
                                               VSNormType.average_anterior_wall,
                                               VSNormField.complete)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_avg_norm_complete_warp_contour(self):

        with open("vs_load_check_cum/expected_cum_norm_avg_norm_complete_warp_contour.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        complete_dfs = self.load_sequences(Path("vs_load_check_cum/df_complete"))
        contour_dfs = self.load_sequences(Path("vs_load_check_cum/df_contour"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               contour_dfs,
                                               complete_dfs,
                                               VSNormType.average_anterior_wall)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_avg_norm_complete_warp_contour.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.contours,
                                               VSNormType.average_anterior_wall,
                                               VSNormField.complete)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")


    def test_cum_vs_norm_vicinity_norm_rest_warp_rest(self):

        with open("vs_load_check_cum/expected_cum_norm_vicinity_norm_rest_warp_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        
        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               VSNormType.contour_vicinity)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_vicinity_norm_rest_warp_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.rest,
                                               VSNormType.contour_vicinity,
                                               VSNormField.rest)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_vicinity_norm_rest_warp_contour(self):

        with open("vs_load_check_cum/expected_cum_norm_vicinity_norm_rest_warp_contour.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        contour_dfs = self.load_sequences(Path("vs_load_check_cum/df_contour"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               contour_dfs,
                                               rest_dfs,
                                               VSNormType.contour_vicinity)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_vicinity_norm_rest_warp_contour.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.contours,
                                               VSNormType.contour_vicinity,
                                               VSNormField.rest)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_vicinity_norm_complete_warp_rest(self):

        with open("vs_load_check_cum/expected_cum_norm_vicinity_norm_complete_warp_rest.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        complete_dfs = self.load_sequences(Path("vs_load_check_cum/df_complete"))
        
        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               rest_dfs,
                                               complete_dfs,
                                               VSNormType.contour_vicinity)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_vicinity_norm_complete_warp_rest.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.rest,
                                               VSNormType.contour_vicinity,
                                               VSNormField.complete)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

    def test_cum_vs_norm_vicinity_norm_complete_warp_contour(self):

        with open("vs_load_check_cum/expected_cum_norm_vicinity_norm_complete_warp_contour.pkl", "r+b") as file:
            expected_vs_data = pickle.load(file)
            expected_x = expected_vs_data["x"]
            expected_y = expected_vs_data["y"]
            expected_vs = expected_vs_data["slide"]

        slice = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/slice.mha"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage("vs_load_check_cum/mask.mha"))

        # With DFs loading
        moving_masks = self.load_sequences(Path("vs_load_check_cum/moving_masks"))
        cavity_dfs = self.load_sequences(Path("vs_load_check_cum/df_cavity"))
        rest_dfs = self.load_sequences(Path("vs_load_check_cum/df_rest"))
        complete_dfs = self.load_sequences(Path("vs_load_check_cum/df_complete"))
        contour_dfs = self.load_sequences(Path("vs_load_check_cum/df_contour"))

        detector = CumulativeVisceralSlideDetectorDF()

        x, y, vs = detector.get_visceral_slide(moving_masks,
                                               cavity_dfs,
                                               rest_dfs,
                                               contour_dfs,
                                               complete_dfs,
                                               VSNormType.contour_vicinity)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

        if plot:
            plt.figure()
            plt.imshow(slice[-2], cmap="gray")
            plt.scatter(x, y, s=5, c=vs, cmap="jet")
            plt.colorbar()
            plt.savefig("vs_load_check_cum/vs_cum_norm_vicinity_norm_complete_warp_contour.png", bbox_inches='tight', pad_inches=0)
            plt.show(bbox_inches='tight', pad_inches=0)

        # With image registration performed on the fly
        detector = CumulativeVisceralSlideDetectorReg()
        
        x, y, vs = detector.get_visceral_slide(slice,
                                               mask,
                                               VSWarpingField.contours,
                                               VSNormType.contour_vicinity,
                                               VSNormField.complete)

        self.assertTrue(np.array_equal(expected_x, x),
                        "Incorrect x coordinates of contour")
        self.assertTrue(np.array_equal(expected_y, y),
                        "Incorrect y coordinates of contour")
        self.assertTrue(np.array_equal(expected_vs, vs),
                        "Incorrect visceral slide")

