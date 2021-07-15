import math
import numpy as np
import pickle
import json
from pathlib import Path
from cinemri.definitions import CineMRISlice
from config import *
from utils import load_visceral_slides, binning_intervals, contour_stat, get_inspexp_frames
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
import SimpleITK as sitk
from vs_definitions import VisceralSlide, Region

def get_regions_expectation(vs_path, vs_range, regions_num=120, heathly_inds=None, plot=False, inspexp_data=None, images_path=None, output_path=None):
    neg_visceral_slides = load_visceral_slides(vs_path)

    # Filter out visceral slides if list of ids is provided
    if heathly_inds:
        neg_visceral_slides = [vs for vs in neg_visceral_slides if vs.patient_id in heathly_inds]

    colors = cm.rainbow(np.linspace(0, 1, regions_num))
    vs_regions = []
    for i in range(len(neg_visceral_slides)):
        vs = neg_visceral_slides[i]
        regions = vs.to_regions(regions_num, colors)
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

        means.append(np.mean(values))
        stds.append(np.mean(values))

    if plot:
        vs = neg_visceral_slides[0]
        slice = CineMRISlice.from_full_id(vs.full_id)

        if inspexp_data is None:
            image = sitk.ReadImage(str(slice.build_path(images_path)))
            frame = sitk.GetArrayFromImage(image)[-2]
        else:
            frame, _ = get_inspexp_frames(slice, inspexp_data, images_path)

        plt.figure()
        plt.imshow(frame, cmap="gray")

        vs_regs = vs.to_regions(regions_num, means)
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
        plt.savefig(output_path / "inspexp_expectation_vi_{}.png".format(regions_num), bbox_inches='tight', pad_inches=0)
        plt.close()

    return {"means": means, "stds": stds}


def gaussian_2d(sigma):
    """
    Paramaters
    ----------
    Input:
    sigma_mm

    Output:
    kernel: kernel
    x : matrix of x coordinates of the filter
    y : matrix of y coordinates of the filter
    """
    kernel_half_size = 3 * sigma

    # number of pixes by x an y axes in grid
    pixels_number = math.ceil(kernel_half_size * 2) + 1
    x, y = np.meshgrid(np.linspace(-kernel_half_size, kernel_half_size, pixels_number),
                       np.linspace(-kernel_half_size, kernel_half_size, pixels_number))
    kernel = np.exp(-((x * x + y * y) / (2.0 * sigma ** 2))) / 2 / np.pi / sigma ** 2
    return kernel, x, y


def get_vs_healthy_expectation(vs_path, heathly_inds, grid_size, plot=False, output_path=None):
    visceral_slides = load_visceral_slides(vs_path)
    neg_visceral_slides = [vs for vs in visceral_slides if vs.patient_id in heathly_inds]
    
    grid = np.zeros(grid_size)
    freq = np.zeros(grid_size)
    bins_y = binning_intervals(0, 1, grid_size[0])
    bins_x = binning_intervals(0, 1, grid_size[1])

    for vs in neg_visceral_slides:
        xs, ys = vs.x, vs.y
        values = vs.values

        # scale x and y to (0,1)
        x_scaled = (xs - vs.origin_x) / vs.width
        y_scaled = (ys - vs.origin_y) / vs.height

        for x, y, value in zip(x_scaled, y_scaled, values):
            diff_x = bins_x - x
            index_x = np.argmin(np.abs(diff_x))

            diff_y = bins_y - y
            index_y = np.argmin(np.abs(diff_y))

            grid[index_y, index_x] += value
            freq[index_y, index_x] += 1

    # For 0 freq the corresponding cum VS values is 0
    freq[freq == 0] = 1
    grid /= freq

    if plot:
        plt.figure()
        plt.imshow(grid, cmap="jet")
        plt.colorbar()
        plt.savefig(output_path / "average.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    size = 5
    kernel = np.ones((size, size)) / size**2
    grid_new = signal.fftconvolve(grid, kernel, mode='same')

    if plot:
        plt.figure()
        plt.imshow(grid_new, cmap="jet")
        plt.colorbar()
        plt.savefig(output_path / "smoothed_uniform.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    kernel2, _, _ = gaussian_2d(1.5)
    expected_vs = signal.fftconvolve(grid, kernel2, mode='same')

    if plot:
        #plt.figure()
        #plt.imshow(kernel2)
        #plt.show(bbox_inches='tight', pad_inches=0)

        plt.figure()
        plt.imshow(expected_vs, cmap="jet")
        plt.colorbar()
        plt.savefig(output_path / "smoothed_gaussian.png", bbox_inches='tight', pad_inches=0)
    
    return expected_vs


def get_avg_contour_size(vs_path, heathly_inds=None):
    neg_visceral_slides = load_visceral_slides(vs_path)

    # Filter out visceral slides if list of ids is provided
    if heathly_inds:
        neg_visceral_slides = [vs for vs in neg_visceral_slides if vs.patient_id in heathly_inds]

    widths = []
    heights = []

    for vs in neg_visceral_slides:
        widths.append(vs.width)
        heights.append(vs.height)

    return round(np.mean(widths)), round(np.mean(heights))


def norm_vs(vs, expected_vs, frame, output_path):
    im_height, im_width = frame.shape
    grid_size = expected_vs.shape

    bins_y = binning_intervals(0, 1, grid_size[0])
    bins_x = binning_intervals(0, 1, grid_size[1])

    xs, ys = vs.x, vs.y
    values = vs.values
    vs_image = np.zeros((im_height, im_width))
    for x, y, v in zip(xs, ys, values):
        vs_image[y, x] = v

    """
    plt.figure()
    plt.imshow(frame, cmap="gray")
    plt.scatter(xs, ys, s=5, c=values, cmap="jet")
    plt.colorbar()
    plt.title(vs.patient_id)
    plt.show(bbox_inches='tight', pad_inches=0)
    """

    # scale x and y to (0,1)
    x_scaled = (xs - vs.origin_x) / vs.width
    y_scaled = (ys - vs.origin_y) / vs.height
    vs_norm = []
    for x, y, value in zip(x_scaled, y_scaled, values):
        diff_x = bins_x - x
        index_x = np.argmin(np.abs(diff_x))

        diff_y = bins_y - y
        index_y = np.argmin(np.abs(diff_y))

        expected_value = expected_vs[index_y, index_x]
        norm_value = (value / expected_value) if (expected_value > 10**(-4)) else 100
        vs_norm.append(norm_value)

    vs_norm = np.array(vs_norm)
    print("{} zero values encoutered".format(sum(vs_norm == 100)))

    vs_norm_image = np.zeros((im_height, im_width))
    for x, y, v in zip(xs, ys, vs_norm):
        vs_norm_image[y, x] = v

    plt.figure()
    plt.imshow(frame, cmap="gray")
    plt.scatter(xs, ys, s=5, c=vs_norm, cmap="jet")
    plt.colorbar()
    #plt.title(vs.patient_id)
    plt.savefig(output_path / "{}.png".format(vs.full_id), bbox_inches='tight', pad_inches=0)
    plt.close()

    return xs, ys, vs_norm

# To remove outliers
def get_vs_range(visceral_slides):
    # Statistics useful for prediction
    all_vs_values = []
    for visceral_slide in visceral_slides:
        all_vs_values.extend(visceral_slide.values)

    vs_abs_max = np.max(all_vs_values)
    vs_abs_min = np.min(all_vs_values)
    vs_q1 = np.quantile(all_vs_values, 0.25)
    vs_q3 = np.quantile(all_vs_values, 0.75)
    vs_iqr = vs_q3 - vs_q1
    vs_min = vs_q1 - 1.5 * vs_iqr
    vs_max = vs_q3 + 1.5 * vs_iqr

    return vs_min, vs_max


if __name__ == '__main__':
    detection_path = Path(DETECTION_PATH)
    images_path = detection_path / IMAGES_FOLDER / CONTROL_FOLDER
    masks_path = detection_path / FULL_SEGMENTATION_FOLDER
    insp_exp_control_path = detection_path / VS_CONTROL_FOLDER / AVG_NORM_FOLDER / INS_EXP_VS_FOLDER
    inspexp_file_path = detection_path / METADATA_FOLDER / INSPEXP_FILE_NAME
    # load inspiration and expiration data
    with open(inspexp_file_path) as inspexp_file:
        inspexp_data = json.load(inspexp_file)

    cum_control_path = detection_path / VS_CONTROL_FOLDER / AVG_NORM_FOLDER / CUMULATIVE_VS_FOLDER

    """
        with open(negative_slices_file) as file:
        lines = file.readlines()
        negative_patients_ids = [line.strip().split(SEPARATOR)[0] for line in lines]
    """

    width, height = get_avg_contour_size(cum_control_path)
    print(width)
    print(height)
    print(width / height)

    avg_contour_len = contour_stat(masks_path)
    point_in_chunk = 5
    chunks_num = round(avg_contour_len / point_in_chunk)

    output_path = Path("corner points v2")
    output_path.mkdir(exist_ok=True)

    # TODO: take 130
    """
    cum_visceral_slides = load_visceral_slides(cum_control_path)
    vs_values = []
    for visceral_slide in cum_visceral_slides:
        vs_values.extend(visceral_slide.values)

    vs_min, vs_max = get_vs_range(cum_visceral_slides)

    expectation = get_regions_expectation(cum_control_path, (vs_min, vs_max), chunks_num, None, True, None, images_path,
                                          output_path)

    with open("cumulative_vs_expectation.pkl", "w+b") as f:
        pickle.dump(expectation, f)

    with open("cumulative_vs_expectation.pkl", "r+b") as file:
        expectation_dict = pickle.load(file)
        means, stds = expectation_dict["means"], expectation_dict["stds"]

    print("done")
    """

    """
    print("Cum VS lower limit {}".format(vs_min))
    print("Cum VS upper limit {}".format(vs_max))

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

    inspexp_visceral_slides = load_visceral_slides(insp_exp_control_path)
    vs_values = []
    for visceral_slide in inspexp_visceral_slides:
        vs_values.extend(visceral_slide.values)

    vs_min, vs_max = get_vs_range(inspexp_visceral_slides)
    expectation = get_regions_expectation(insp_exp_control_path, (vs_min, vs_max), chunks_num, None, True, inspexp_data, images_path,
                                          output_path)

    with open("insexp_vs_expectation.pkl", "w+b") as f:
        pickle.dump(expectation, f)

    with open("insexp_vs_expectation.pkl", "r+b") as file:
        expectation_dict = pickle.load(file)
        means, stds = expectation_dict["means"], expectation_dict["stds"]

    print("done")

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

    #for i in range(len(visceral_slides)):
    """
        for i in range(1):
        vs = visceral_slides[i]
        norm_vs_chunks(vs, means, stds, images_path, True, output_path)
    """


    """
    prior = get_vs_healthy_prior(cumulative_vs_path, negative_patients_ids, (height, width), True, output_path)

    visceral_slides = load_visceral_slides(cumulative_vs_path)
    pos_visceral_slides = [vs for vs in visceral_slides if vs.patient_id not in negative_patients_ids]

    for i in range(len(pos_visceral_slides)):
        vs = pos_visceral_slides[i]
        slice = CineMRISlice.from_full_id(vs.full_id)
        image = sitk.ReadImage(str(slice.build_path(images_path)))
        frame = sitk.GetArrayFromImage(image)[-2]
        norm_vs(vs, prior, frame, output_path)
    """
