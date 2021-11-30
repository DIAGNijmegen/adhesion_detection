import numpy as np
from pathlib import Path
from enum import Enum, unique
from cinemri.contour import Contour
from .config import *


@unique
class VSExpectationNormType(Enum):
    # Division by mean
    mean_div = 0
    # Standardize with std
    standardize = 1


# Transformation to apply to visceral slide values
@unique
class VSTransform(Enum):
    none = 0
    log = 1
    sqrt = 2


class Region:
    """An object representing a coordinates region with a value at a specific coordinate"""

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
           A standard deviation of the region
        """
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.values = np.atleast_1d(values)
        self.mean = mean
        self.std = std
        self.points = np.column_stack((self.x, self.y, self.values))

    def __len__(self):
        return len(self.x)

    @classmethod
    def from_point(cls, point):
        """
        Initialises a Region object from a single point
        Parameters
        ----------
        point : tuple
           (x, y, value) - the coordinates of the point and the value at this point

        Returns
        -------
        region : Region
           A region that consists of a single point
        """
        x, y, value = point[0], point[1], point[2]
        region = cls(x, y, value)
        return region

    @classmethod
    def from_points(cls, points):
        """
        Initialises a Region object from points
        Parameters
        ----------
        points : ndarray
            An array of points coordinates and values at a point. The first column - coordinates by x axis,
            the second - coordinates by y axis, the third - values

        Returns
        -------
        region : Region
            A region that consists of a single point
        """
        x, y, values = points[:, 0], points[:, 1], points[:, 2]
        region = cls(x, y, values)
        return region

    def append(self, x, y, value):
        """
        Appends one coordinate and its value to the region
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

        self.points = np.column_stack((self.x, self.y, self.values))

    def append_point(self, point):
        """
        Appends one coordinate and its values to the region
        Parameters
        ----------
        point : tuple of (int, int, float)
           A coordinate and the corresponding value to append
        value : float
           The corresponding value
        """
        self.x = np.concatenate((self.x, [point[0]]))
        self.y = np.concatenate((self.y, [point[1]]))
        self.values = np.concatenate((self.values, [point[2]]))

        self.points = np.column_stack((self.x, self.y, self.values))

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

        self.points = np.column_stack((self.x, self.y, self.values))

    @property
    def size(self):
        """Size of a rectangle that encloses the region"""
        if len(self.x) == 0:
            return 0, 0

        def compute_len(axis=0):
            values = [point[axis] for point in self.points]
            length = np.max(values) - np.min(values) + 1
            return length

        width = compute_len(axis=0)
        height = compute_len(axis=1)
        return width, height

    def exceeded_size(self, size):
        """
        Checks if the size of a rectangle that encloses the region is larger than a given size
        Parameters
        ----------
        size : tuple
           A size to compare with

        Returns
        -------
        flag: bool
           A boolean flag indicating whether region size is larger than the given size
        """
        width, height = self.size
        return width >= size[0] or height >= size[1]


class VisceralSlide(Contour):
    """An object representing visceral slide for a Cine-MRI slice"""

    def __init__(self, patient_id, study_id, slice_id, visceral_slide_data):
        """
        Parameters
        ----------
        patient_id : str
           An id of a patient a Cine-MRI slice belongs to
        study_id : str
           An id of a study a Cine-MRI slice belongs to
        slice_id : str
           An id of a Cine-MRI slice
        visceral_slide_data : dict
           A dictionary containing the coordinates of abdominal cavity contour
           and visceral slide value at each coordinate
        """

        super().__init__(visceral_slide_data["x"], visceral_slide_data["y"])

        self.values = np.array(visceral_slide_data["slide"])
        self.patient_id = patient_id
        self.study_id = study_id
        self.slice_id = slice_id
        self.full_id = SEPARATOR.join([patient_id, study_id, slice_id])

    def zeros_fix(self):
        """Replace zeros with highest non 0 in visceral slide values"""
        zero_placeholder = np.min([value for value in self.values if value > 0])
        self.values = np.array(
            [value if value > 0 else zero_placeholder for value in self.values]
        )

    def to_regions(self, means=None, stds=None):
        """
        Splits visceral slide into chunks starting from the bottom left point of the contour
        in the clock-wise direction. The provided mean and stds should correspond to these chunks

        Parameters
        ----------
        means : list of float, optional
           A list of visceral slide mean by chunk
        stds : list of float, optional
           A list of visceral slide standard deviation by chunk

        Returns
        -------
        regions : list of Regions
           A list of objects of Region type that represent chunks of visceral slide map
           and include mean and standard deviation of visceral slide value in each chunk
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

            mean = means[i] if means is not None else None
            std = stds[i] if stds is not None else None

            # Normal situation, take the connected region
            if reg_start < reg_end:
                x_reg = xs[reg_start:reg_end]
                y_reg = ys[reg_start:reg_end]
                val_reg = values[reg_start:reg_end]

                region = Region(x_reg, y_reg, val_reg, mean, std)
            else:
                # We went beyond the fist contour coordinate, so add up a region from the region start till
                # the end of contour and from the start of contour till the region end
                x_reg = xs[reg_start:]
                y_reg = ys[reg_start:]
                val_reg = values[reg_start:]
                region = Region(x_reg, y_reg, val_reg, mean, std)
                region.extend(xs[:reg_end], ys[:reg_end], values[:reg_end])

            regions.append(region)
            reg_start = reg_end
            if i < regions_num - 1:
                if i == regions_num - 2:
                    reg_end = last_ind
                else:
                    reg_end += chunk_lens[i + 1]

        return regions

    def norm_with_expectation(
        self, means, stds, expectation_norm_type=VSExpectationNormType.mean_div
    ):
        """Normalises visceral slide by provided means and standard deviations assuming that
        means and standard deviations correspond to regions obtained with to_regions method

        Parameters
        ----------
        means, stds : list of float
           A list of visceral slide mean by chunk
        stds : list of float
           A list of visceral slide standard deviation by chunk
        expectation_norm_type : VSExpectationNormType, default=VSExpectationNormType.mean_div
           A type of normalisation to apply
        """

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
        """
        Build a path to a folder that contains visceral slide assuming standard folders hierarchy
        Parameters
        ----------
        relative_path : Path
            A relative path to locate a slice file
        Returns
        -------
        path : Path
            A path to a folder that contains visceral slide
        """
        return (
            Path(relative_path)
            / self.patient_id
            / self.study_id
            / (self.slice_id + extension)
        )
