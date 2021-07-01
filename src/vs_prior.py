import numpy as np
from pathlib import Path
from cinemri.config import ARCHIVE_PATH
from cinemri.definitions import CineMRISlice
from config import METADATA_FOLDER, NEGATIVE_SLICES_FILE_NAME, SEPARATOR
from utils import load_visceral_slides, binning_intervals
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
import SimpleITK as sitk
import math
from cinemri.contour import get_abdominal_contour_top, _get_tangent_vectors, get_most_vertical_point


class Region:
    """An object representing visceral slide for a Cine-MRI slice
    """

    def __init__(self, x, y, values, col):
        self.x = x
        self.y = y
        self.values = values
        self.col = col

    def extend(self, x, y, values):
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        self.values = np.concatenate((self.values, values))


def vs_to_regions(vs, regions_num, colors):
    xs, ys = vs.x, vs.y
    values = vs.values

    vs_len = len(vs.values)
    indices = np.arange(vs_len)
    chunks = np.array_split(indices, regions_num)
    chunk_lens = np.array([len(chunk) for chunk in chunks])
    np.random.shuffle(chunk_lens)

    x_bottom_left, y_bottom_left = vs.bottom_left_point
    coords = np.column_stack((xs, ys))
    ind = np.where((coords == (x_bottom_left, y_bottom_left)).all(axis=1))[0][0]

    reg_start = ind
    reg_end = reg_start + chunk_lens[0]
    last_ind = reg_start

    regions = []
    for i in range(regions_num):
        if reg_end >= vs_len:
            reg_end -= vs_len

        if reg_start >= vs_len:
            reg_start -= vs_len

        # Normal situation, take the connected region
        if reg_start < reg_end:
            x_reg = xs[reg_start:reg_end]
            y_reg = ys[reg_start:reg_end]
            val_reg = values[reg_start:reg_end]
            region = Region(x_reg, y_reg, val_reg, colors[i])
        else:
            x_reg1 = xs[reg_start:]
            y_reg1 = ys[reg_start:]
            val_reg1 = values[reg_start:]
            region = Region(x_reg1, y_reg1, val_reg1, colors[i])
            region.extend(xs[:reg_end], ys[:reg_end], values[:reg_end])

        regions.append(region)
        reg_start = reg_end
        if i < regions_num - 1:
            if i == regions_num - 2:
                reg_end = last_ind
            else:
                reg_end += chunk_lens[i+1]

    return regions


def get_regions_prior(vs_path, heathly_inds, regions_num=120, plot=False, images_path=None, output_path=None):
    visceral_slides = load_visceral_slides(vs_path)
    neg_visceral_slides = [vs for vs in visceral_slides if vs.patient_id in heathly_inds]

    colors = cm.rainbow(np.linspace(0, 1, regions_num))
    vs_regions = []
    for i in range(len(neg_visceral_slides)):
        vs = neg_visceral_slides[i]
        regions = vs_to_regions(vs, regions_num, colors)
        vs_regions.append(regions)

    vs_regions = np.array(vs_regions)
    # Now for each region calculate the average
    averages = []
    for i in range(regions_num):
        cur_regions = vs_regions[:, i]
        values = cur_regions[0].values
        for j in range(1, len(neg_visceral_slides)):
            values = np.concatenate((values, cur_regions[j].values))
        avg = values.mean()
        averages.append(avg)

    if plot:
        vs = neg_visceral_slides[0]
        slice = CineMRISlice.from_full_id(vs.full_id)
        image = sitk.ReadImage(str(slice.build_path(images_path)))
        frame = sitk.GetArrayFromImage(image)[-2]

        plt.figure()
        plt.imshow(frame, cmap="gray")

        vs_regs = vs_to_regions(vs, regions_num, averages)
        reg = vs_regs[0]
        x = reg.x
        y = reg.y
        c = np.ones(len(reg.x)) * reg.col
        for i in range(1, len(vs_regs)):
            reg = vs_regs[i]
            x = np.concatenate((x, reg.x))
            y = np.concatenate((y, reg.y))
            c = np.concatenate((c, np.ones(len(reg.x)) * reg.col))
        plt.scatter(x, y, s=5, c=c)
        plt.colorbar()
        plt.savefig(output_path / "prior_vi_{}.png".format(regions_num), bbox_inches='tight', pad_inches=0)
        plt.close()

    return averages


def norm_vs_chunks(vs, prior, images_path, plot=False, output_path=None):
    regions_num = len(prior)
    vs_regs = vs_to_regions(vs, regions_num, prior)

    if plot:
        slice = CineMRISlice.from_full_id(vs.full_id)
        image = sitk.ReadImage(str(slice.build_path(images_path)))
        frame = sitk.GetArrayFromImage(image)[-2]
        
        plt.figure()
        plt.imshow(frame, cmap="gray")
    
        reg = vs_regs[0]
        x = reg.x
        y = reg.y
        c = reg.values / reg.col
        for i in range(1, len(vs_regs)):
            reg = vs_regs[i]
            x = np.concatenate((x, reg.x))
            y = np.concatenate((y, reg.y))
            c = np.concatenate((c, reg.values / reg.col))
        plt.scatter(x, y, s=5, c=c, cmap="jet")
        plt.colorbar()
        #plt.title(vs.patient_id)
        plt.savefig(output_path / "{}.png".format(vs.full_id), bbox_inches='tight', pad_inches=0)
        plt.close()
    
    return vs_regs


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


def get_vs_healthy_prior(vs_path, heathly_inds, grid_size, plot=False, output_path=None):
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


def get_avg_contour_size(vs_path, heathly_inds):
    visceral_slides = load_visceral_slides(vs_path)
    neg_visceral_slides = [vs for vs in visceral_slides if vs.patient_id in heathly_inds]

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



if __name__ == '__main__':
    archive_path = ARCHIVE_PATH
    images_path = archive_path / "detection_new" / "images"
    negative_slices_file = archive_path / METADATA_FOLDER / NEGATIVE_SLICES_FILE_NAME
    cumulative_vs_path = Path("../../data/vs_cum/cumulative_vs_contour_reg_det_full_df")

    with open(negative_slices_file) as file:
        lines = file.readlines()
        negative_patients_ids = [line.strip().split(SEPARATOR)[0] for line in lines]

    width, height = get_avg_contour_size(cumulative_vs_path, negative_patients_ids)
    print(width)
    print(height)
    print(width / height)

    output_path = Path("corner points v2")
    output_path.mkdir(exist_ok=True)

    # TODO: take 130
    prior = get_regions_prior(cumulative_vs_path, negative_patients_ids, 65, True, images_path, output_path)
    get_regions_prior(cumulative_vs_path, negative_patients_ids, 130, True, images_path, output_path)
    get_regions_prior(cumulative_vs_path, negative_patients_ids, 260, True, images_path, output_path)
    get_regions_prior(cumulative_vs_path, negative_patients_ids, 326, True, images_path, output_path)

    """
    visceral_slides = load_visceral_slides(cumulative_vs_path)
    for i in range(len(visceral_slides)):
        vs = visceral_slides[i]
        vs_image = vs.build_path(images_path)
        image = sitk.ReadImage(str(vs_image))
        frame = sitk.GetArrayFromImage(image)[-2]

        x1, y1 = vs.top_left_point
        x2, y2 = vs.top_right_point
        x3, y3 = vs.bottom_left_point
        x4, y4 = vs.bottom_right_point

        plt.figure()
        plt.imshow(frame, cmap="gray")
        plt.scatter(vs.x, vs.y, s=5, color="b")
        plt.scatter(x1, y1, s=10, color="r")
        plt.scatter(x2, y2, s=10, color="y")
        plt.scatter(x3, y3, s=10, color="g")
        plt.scatter(x4, y4, s=10, color="c")
        plt.savefig(output_path / "{}.png".format(vs.full_id), bbox_inches='tight', pad_inches=0)
        plt.close()
    """


    """
    visceral_slides = load_visceral_slides(cumulative_vs_path)
    pos_visceral_slides = [vs for vs in visceral_slides if vs.patient_id not in negative_patients_ids]

    for i in range(len(pos_visceral_slides)):
        vs = pos_visceral_slides[i]
        norm_vs_chunks(vs, prior, images_path, True, output_path)
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
