import math
import numpy as np
import pickle
import json
from pathlib import Path
from cinemri.definitions import CineMRISlice
from config import *
from utils import load_visceral_slides, binning_intervals, contour_stat, get_inspexp_frames, get_avg_contour_size, \
    get_vs_range
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimpleITK as sitk
from detection_pipeline import vs_values_boxplot, VSTransform


def get_regions_expectation(vs_path, vs_range, regions_num=120, transform=VSTransform.none, heathly_inds=None, plot=False, inspexp_data=None, images_path=None, output_path=None):
    neg_visceral_slides = load_visceral_slides(vs_path)

    # Filter out visceral slides if list of ids is provided
    if heathly_inds:
        neg_visceral_slides = [vs for vs in neg_visceral_slides if vs.patient_id in heathly_inds]

    colors = cm.rainbow(np.linspace(0, 1, regions_num))
    vs_regions = []
    for i in range(len(neg_visceral_slides)):
        vs = neg_visceral_slides[i]
        regions = vs.to_regions(colors)
        vs_regions.append(regions)

    vs_regions = np.array(vs_regions)
    # Now for each region calculate the average
    means = []
    stds = []
    for i in range(regions_num):
        cur_regions = vs_regions[:, i]
        values = cur_regions[0].values
        for j in range(1, len(neg_visceral_slides)):
            values = np.concatenate((values, cur_regions[j].values))

        # Remove the outliers
        values = np.array([value for value in values if vs_range[0] <= value <= vs_range[1]])

        if transform == VSTransform.log:
            values = np.log(values)
        elif transform == VSTransform.sqrt:
            values = np.sqrt(values)

        means.append(np.mean(values))
        stds.append(np.std(values))

    if plot:
        vs = neg_visceral_slides[0]
        slice = CineMRISlice.from_full_id(vs.full_id)

        title = "inspexp_expectation_vi" if inspexp_data is None else "cumulative_vi"
        if transform == VSTransform.log:
            title += "_log"
        elif transform == VSTransform.sqrt:
            title += "_sqrt"

        if inspexp_data is None:
            image = sitk.ReadImage(str(slice.build_path(images_path)))
            frame = sitk.GetArrayFromImage(image)[-2]
        else:
            frame, _ = get_inspexp_frames(slice, inspexp_data, images_path)

        plt.figure()
        plt.imshow(frame, cmap="gray")

        vs_regs = vs.to_regions(means)
        reg = vs_regs[0]
        x = reg.x
        y = reg.y
        c = np.ones(len(reg.x)) * reg.mean
        for i in range(1, len(vs_regs)):
            reg = vs_regs[i]
            x = np.concatenate((x, reg.x))
            y = np.concatenate((y, reg.y))
            c = np.concatenate((c, np.ones(len(reg.x)) * reg.mean))
        plt.scatter(x, y, s=5, c=c)
        plt.colorbar()
        plt.savefig(output_path / "{}_{}.png".format(title, regions_num), bbox_inches='tight', pad_inches=0)
        plt.close()

    return {"means": means, "stds": stds}


if __name__ == '__main__':
    detection_path = Path(DETECTION_PATH)
    images_path = detection_path / IMAGES_FOLDER / CONTROL_FOLDER
    masks_path = detection_path / FULL_SEGMENTATION_FOLDER
    insp_exp_control_path = detection_path / VS_CONTROL_FOLDER / VICINITY_NORM_FOLDER / INS_EXP_VS_FOLDER
    inspexp_file_path = detection_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    output_path = detection_path / "control_vs_stat_vicinity"
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    cum_control_path = detection_path / VS_CONTROL_FOLDER / VICINITY_NORM_FOLDER / CUMULATIVE_VS_FOLDER

    cumulative_vs = True
    transform = VSTransform.sqrt
    plot_distr = False
    compute_stat = True

    vs_path = cum_control_path if cumulative_vs else insp_exp_control_path
    output_path = output_path / "cumulative" if cumulative_vs else output_path / "inspexp"

    visceral_slides = load_visceral_slides(vs_path)
    vs_min, vs_max = get_vs_range(visceral_slides, False)

    print("Cum VS lower limit {}".format(vs_min))
    print("Cum VS upper limit {}".format(vs_max))

    if plot_distr:
        # Distribution for all
        vs_values_boxplot(visceral_slides, output_path)
        # Distribution for prior region
        vs_values_boxplot(visceral_slides, output_path, prior_only=True)
        # Distribution for all outliers removed
        vs_values_boxplot(visceral_slides, output_path, vs_min, vs_max)
        # Distribution for prior region outliers removed
        vs_values_boxplot(visceral_slides, output_path, vs_min, vs_max, prior_only=True)
        # Distribution for all sqrt transfrom
        vs_values_boxplot(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.sqrt)
        # Distribution for prior region sqrt transfrom
        vs_values_boxplot(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.sqrt, prior_only=True)
        # Distribution for all sqrt transfrom
        vs_values_boxplot(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.log)
        # Distribution for prior region sqrt transfrom
        vs_values_boxplot(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.log, prior_only=True)

    if compute_stat:
        avg_contour_len = contour_stat(masks_path)
        point_in_chunk = 5
        chunks_num = round(avg_contour_len / point_in_chunk)

        output_path = Path("vs_expectation")
        output_path.mkdir(exist_ok=True)

        vs_min, vs_max = get_vs_range(visceral_slides, False)

        if cumulative_vs:
            inspexp_data = None

        expectation = get_regions_expectation(vs_path, (vs_min, vs_max), chunks_num, transform, None, True, inspexp_data, images_path,
                                              output_path)


        pkl_title = "cumulative_vs_expectation_vicinity" if cumulative_vs else "inspexp_vs_expectation_vicinity"
        if transform == VSTransform.log:
            pkl_title += "_log"
        elif transform == VSTransform.sqrt:
            pkl_title += "_sqrt"

        with open(pkl_title + ".pkl", "w+b") as f:
            pickle.dump(expectation, f)

        with open(pkl_title + ".pkl", "r+b") as file:
            expectation_dict = pickle.load(file)
            means, stds = expectation_dict["means"], expectation_dict["stds"]

    """
    vs_values = [vs for vs in vs_values if vs_min <= vs <= vs_max]
    vs_values_log = np.log(vs_values)

    plt.figure()
    plt.boxplot(vs_values_log)
    plt.savefig("vs_boxplot_cum_control_outl_removed_log", bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure()
    plt.hist(vs_values_log, bins=200)
    plt.savefig("vs_hist_cum_control_outl_removed_log", bbox_inches='tight', pad_inches=0)
    plt.show()
    """

    """
    inspexp_visceral_slides = load_visceral_slides(insp_exp_control_path)
    vs_values = []
    for visceral_slide in inspexp_visceral_slides:
        vs_values.extend(visceral_slide.values)

    vs_min, vs_max = get_vs_range(inspexp_visceral_slides, False)
    expectation = get_regions_expectation(insp_exp_control_path, (vs_min, vs_max), chunks_num, VSTransform.sqrt, None, True, inspexp_data, images_path,
                                          output_path)

    with open("insexp_vs_expectation.pkl", "w+b") as f:
        pickle.dump(expectation, f)

    with open("insexp_vs_expectation.pkl", "r+b") as file:
        expectation_dict = pickle.load(file)
        means, stds = expectation_dict["means"], expectation_dict["stds"]

    print("done")
    """


    """
    print("Inspexp VS lower limit {}".format(vs_min))
    print("Inspexp VS upper limit {}".format(vs_max))

    vs_values = [vs for vs in vs_values if 0 < vs <= vs_max]
    vs_values_log = np.sqrt(vs_values)

    plt.figure()
    plt.boxplot(vs_values_log)
    plt.savefig("vs_boxplot_inspexp_control_outl_removed_sqrt", bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure()
    plt.hist(vs_values_log, bins=200)
    plt.savefig("vs_hist_inspexp_control_outl_removed_sqrt", bbox_inches='tight', pad_inches=0)
    plt.show()
    """

    """
    expectation = get_regions_expectation(cum_control_path, (vs_min, vs_max), chunks_num, None, True, None, images_path, output_path)

    with open("cumulative_vs_expectation.pkl", "w+b") as f:
        pickle.dump(expectation, f)

    with open("cumulative_vs_expectation.pkl", "r+b") as file:
        expectation_dict = pickle.load(file)
        means, stds = expectation_dict["means"], expectation_dict["stds"]

    print("done")

    visceral_slides = load_visceral_slides(cum_control_path)
    """