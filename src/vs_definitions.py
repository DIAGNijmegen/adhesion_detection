import numpy as np
from pathlib import Path
from enum import Enum, unique
from cinemri.contour import Contour
from config import *

@unique
class VSExpectationNormType(Enum):
    # Division by mean
    mean_div = 0
    # Standardize with std
    standardize = 1


class Region:
    """An object representing a coordinates region with a value at a specific coordinate
    """

    def __init__(self, x, y, values, mean=None, std=None):
        """

        Parameters
        ----------
        x, y : list of int or int
           The coordinates of the region
        values : list of float or float
           The values that correspond to coordinates
        mean : float
           A mean of the region
        std : float
           A standartd deviation of the region
        """
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.values = np.atleast_1d(values)
        self.mean = mean
        self.std = std

    def append(self, x, y, value):
        """
        Appends one coordinate and its values to the region
        Parameters
        ----------
        x, y : int
           A coordinate to append
        value : float
           The corresponding value
        """
        self.x = np.concatenate((self.x, [x]))
        self.y = np.concatenate((self.y, [y]))
        self.values = np.concatenate((self.values, [value]))

    def extend(self, x, y, values):
        """
        Appends one coordinate and its values to the region
        Parameters
        ----------
        x, y : list of int
            Coordinates to extend with
        values : list of floats
            Corresponding values
        """
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        self.values = np.concatenate((self.values, values))


# TODO: moveis_slice_vs_suitable
class VisceralSlide(Contour):
    """An object representing visceral slide for a Cine-MRI slice
    """

    def __init__(self, patient_id, study_id, slice_id, visceral_slide_data):

        super().__init__(visceral_slide_data["x"], visceral_slide_data["y"])

        self.values = np.array(visceral_slide_data["slide"])
        self.patient_id = patient_id
        self.study_id = study_id
        self.slice_id = slice_id
        self.full_id = SEPARATOR.join([patient_id, study_id, slice_id])

    def to_regions(self, means, stds=None):
        """
        Splits visceral slide into chunks starting from the bottom left point of the contour
        in the clock-wise direction. The provided mean and stds should correspond to these chunks
        """
        regions_num = len(means)

        xs, ys = self.x, self.y
        values = self.values

        vs_len = len(values)
        indices = np.arange(vs_len)
        chunks = np.array_split(indices, regions_num)
        chunk_lens = np.array([len(chunk) for chunk in chunks])
        np.random.shuffle(chunk_lens)

        x_bottom_left, y_bottom_left = self.bottom_left_point
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
                region = Region(x_reg, y_reg, val_reg, means[i], stds[i]) if stds is not None\
                    else Region(x_reg, y_reg, val_reg, means[i])
            else:
                # We went beyond the fist contour coordinate, so add up a region from the region start till
                # the end of contour and from the start of contour till the region end
                x_reg = xs[reg_start:]
                y_reg = ys[reg_start:]
                val_reg = values[reg_start:]
                region = Region(x_reg, y_reg, val_reg, means[i], stds[i]) if stds is not None\
                    else Region(x_reg, y_reg, val_reg, means[i])
                region.extend(xs[:reg_end], ys[:reg_end], values[:reg_end])

            regions.append(region)
            reg_start = reg_end
            if i < regions_num - 1:
                if i == regions_num - 2:
                    reg_end = last_ind
                else:
                    reg_end += chunk_lens[i + 1]

        return regions

    def norm_with_expectation(self, means, stds, expectation_norm_type=VSExpectationNormType.mean_div):
        """Normalises visceral slide by provided means and standard deviations assuming that
        means and standard deviations correspond to regions obtained with to_regions method"""
        vs_regs = self.to_regions(means, stds)

        reg = vs_regs[0]
        if expectation_norm_type == VSExpectationNormType.mean_div:
            values = reg.values / reg.mean
        else:
            values = (reg.values - reg.mean) / reg.std

        for i in range(1, len(vs_regs)):
            reg = vs_regs[i]
            if expectation_norm_type == VSExpectationNormType.mean_div:
                vs_norm = reg.values / reg.mean
            else:
                vs_norm = (reg.values - reg.mean) / reg.std

            values = np.concatenate((values, vs_norm))

        self.values = values

    def build_path(self, relative_path, extension=".mha"):
        return Path(relative_path) / self.patient_id / self.study_id / (self.slice_id + extension)
