import numpy as np
from pathlib import Path
from enum import Enum, unique
from config import *

@unique
class VSExpectationNormType(Enum):
    # Division by mean
    mean_div = 0
    # Standardize with std
    standardize = 1


class Region:
    """An object representing visceral slide for a Cine-MRI slice
    """

    def __init__(self, x, y, values, mean, std=None):
        self.x = x
        self.y = y
        self.values = values
        self.mean = mean
        self.std = std

    def extend(self, x, y, values):
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        self.values = np.concatenate((self.values, values))


# TODO: moveis_slice_vs_suitable
class VisceralSlide:
    """An object representing visceral slide for a Cine-MRI slice
    """

    def __init__(self, patient_id, study_id, slice_id, visceral_slide_data):
        self.patient_id = patient_id
        self.study_id = study_id
        self.slice_id = slice_id
        self.full_id = SEPARATOR.join([patient_id, study_id, slice_id])
        self.x = np.array(visceral_slide_data["x"])
        self.y = np.array(visceral_slide_data["y"])
        self.values = np.array(visceral_slide_data["slide"])
        self.origin_x = np.min(self.x)
        self.origin_y = np.min(self.y)
        self.width = np.max(self.x) - self.origin_x
        self.height = np.max(self.y) - self.origin_y
        self.middle_x = self.origin_x + round(self.width / 2)
        self.middle_y = self.origin_y + round(self.height / 2)

        self.__top_coords = None
        self.__bottom_coords = None

        self.__top_left_coords = None
        self.__top_right_coords = None
        self.__bottom_left_coords = None
        self.__bottom_right_coords = None

        self.__bottom_left_point = None
        self.__top_left_point = None
        self.__bottom_right_point = None
        self.__top_right_point = None

    def to_regions(self, regions_num, means, stds=None):
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
                x_reg1 = xs[reg_start:]
                y_reg1 = ys[reg_start:]
                val_reg1 = values[reg_start:]
                region = Region(x_reg1, y_reg1, val_reg1, means[i], stds[i]) if stds is not None\
                    else Region(x_reg1, y_reg1, val_reg1, means[i])
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
        regions_num = len(means)
        vs_regs = self.to_regions(regions_num, means, stds)

        reg = vs_regs[0]
        x = reg.x
        y = reg.y
        if expectation_norm_type == VSExpectationNormType.mean_div:
            values = reg.values / reg.mean
        else:
            values = (reg.values - reg.mean) / reg.std

        for i in range(1, len(vs_regs)):
            reg = vs_regs[i]
            x = np.concatenate((x, reg.x))
            y = np.concatenate((y, reg.y))
            if expectation_norm_type == VSExpectationNormType.mean_div:
                vs_norm = reg.values / reg.mean
            else:
                vs_norm = (reg.values - reg.mean) / reg.std

            values = np.concatenate((values, vs_norm))

        self.x = x
        self.y = y
        self.values = values

    @property
    def top_coords(self):
        if self.__top_coords is None:
            coords = np.column_stack((self.x, self.y))
            self.__top_coords = np.array([coord for coord in coords if coord[1] < self.middle_y])

        return self.__top_coords

    @property
    def bottom_coords(self):
        if self.__bottom_coords is None:
            coords = np.column_stack((self.x, self.y))
            self.__bottom_coords = np.array([coord for coord in coords if coord[1] >= self.middle_y])

        return self.__bottom_coords

    @property
    def top_middle_x(self):
        top_coords_x = self.top_coords[:, 0]
        return top_coords_x.min() + (top_coords_x.max() - top_coords_x.min()) / 2

    @property
    def bottom_middle_x(self):
        bottom_coords_x = self.bottom_coords[:, 0]
        return bottom_coords_x.min() + (bottom_coords_x.max() - bottom_coords_x.min()) / 2

    @property
    def top_left_coords(self):
        if self.__top_left_coords is None:
            top_left_coords = np.array([coord for coord in self.top_coords if coord[0] < self.top_middle_x])
            self.__top_left_coords = top_left_coords[:, 0], top_left_coords[:, 1]

        return self.__top_left_coords

    @property
    def top_right_coords(self):
        if self.__top_right_coords is None:
            top_right_coords = np.array([coord for coord in self.top_coords if coord[0] >= self.top_middle_x])
            self.__top_right_coords = top_right_coords[:, 0], top_right_coords[:, 1]

        return self.__top_right_coords

    @property
    def bottom_left_coords(self):
        if self.__bottom_left_coords is None:
            bottom_left_coords = np.array([coord for coord in self.bottom_coords if coord[0] < self.bottom_middle_x])
            self.__bottom_left_coords = bottom_left_coords[:, 0], bottom_left_coords[:, 1]

        return self.__bottom_left_coords

    @property
    def bottom_right_coords(self):
        if self.__bottom_right_coords is None:
            bottom_right_coords = np.array([coord for coord in self.bottom_coords if coord[0] >= self.bottom_middle_x])
            self.__bottom_right_coords = bottom_right_coords[:, 0], bottom_right_coords[:, 1]

        return self.__bottom_right_coords

    @property
    def bottom_left_point(self):
        x, y = self.bottom_left_coords
        x, y = x.astype(np.float64), y.astype(np.float64)
        bottom_left_x = x.min()
        bottom_left_y = y.max()

        diff = np.sqrt((x - bottom_left_x) ** 2 + (y - bottom_left_y) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    @property
    def top_left_point(self):
        x, y = self.top_left_coords
        x, y = x.astype(np.float64), y.astype(np.float64)

        diff = np.sqrt((x - x.min()) ** 2 + (y - y.min()) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    @property
    def bottom_right_point(self):
        x, y = self.bottom_right_coords
        x, y = x.astype(np.float64), y.astype(np.float64)

        diff = np.sqrt((x - x.max()) ** 2 + (y - y.max()) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    @property
    def top_right_point(self):
        x, y = self.top_right_coords
        x, y = x.astype(np.float64), y.astype(np.float64)

        diff = np.sqrt((x - x.max()) ** 2 + (y - y.min()) ** 2)
        index = np.argmin(diff)
        return x[index], y[index]

    def build_path(self, relative_path, extension=".mha"):
        return Path(relative_path) / self.patient_id / self.study_id / (self.slice_id + extension)
