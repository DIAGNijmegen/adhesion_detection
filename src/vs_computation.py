import numpy as np
import ants
from cinemri.registration import Registrator
from cinemri.contour import Contour, get_anterior_wall_data, mask_to_contour
from cinemri.utils import numpy_2d_to_ants, average_by_vicinity, plot_vs_on_frame
from enum import Enum, unique
import pickle


@unique
class VSNormType(Enum):
    none = 0
    # normalise by the average motion along the anterior abdominal wall
    average_anterior_wall = 1
    # normalize by average motion in the vicinity of each point
    contour_vicinity = 2


@unique
class VSNormField(Enum):
    # use the masked deformation field of the abdominal cavity surroundings
    rest = 0
    # use the entire deformation filed between two frames
    complete = 1


@unique
class VSWarpingField(Enum):
    # use the masked deformation field of the abdominal cavity surroundings
    rest = 0
    # use deformation filed between abdominal cavity contours
    contours = 1


class VisceralSlideDetector:
    @staticmethod
    def calculate_visceral_slide(deformation_field, contour):
        """
        Calculates visceral slide along the specified contour given the specified deformation field

        Parameters
        ----------
        deformation_field : ndarray
           The 2-channels deformation field
        contour : Contour
           The the contour along which to calculate visceral slide

        Returns
        -------
        visceral_slide : ndarray
           Visceral slide along the specified contour given the specified deformation field

        """

        # Get contour and normal neighbours
        uv = np.column_stack((contour.u, contour.v))

        inner_deformation = deformation_field[contour.y, contour.x]
        inner_slide = (uv * inner_deformation).sum(1)

        outer_deformation = deformation_field[contour.y_neighbour, contour.x_neighbour]
        outer_slide = (uv * outer_deformation).sum(1)

        # When (x, y) == (x_neighbour, y_neighbour) it means that (x, y) is located
        # at the border of a frame and a real (x_neighbour, y_neighbour) is outside
        # of the frame. In this case the inner and the outer deformation is the same
        # and the visceral slide will be underestimated.
        # Hence to mitigate it, we set the outer slide to 0 for such point
        # TODO: think if an option better than this workaround is possible, e.g nearest-neighbour like approach
        inner_coord = np.column_stack((contour.x, contour.y))
        outer_coord = np.column_stack((contour.x_neighbour, contour.y_neighbour))
        coords_equal_inds = np.where((inner_coord == outer_coord).all(axis=1))[0]
        outer_slide[coords_equal_inds] = 0

        visceral_slide = inner_slide - outer_slide
        return visceral_slide

    @staticmethod
    def get_motion(deformation_field, x, y, u, v):
        """
        Gets motion along the (x, y) contour in direction specified by (u, v)
        Parameters
        ----------
        deformation_field : ndarray
            A deformation filed describing the motion
        x, y : list of int
            The coordinates specifying the contour along which to calculate the motion
        u, v : list if int
            The components of vectors corresponding to each (x, y) in the direction of which to calculate the motion

        Returns
        -------
        motion : ndarray of float
            The motion along the the (x, y) contour in the (u, v) direction
        """
        motion = []
        for i in range(len(x)):
            deformation = deformation_field[y[i], x[i]]
            current_motion = np.abs(deformation.dot([u[i], v[i]]))
            motion.append(current_motion)

        # Replace zeros with highest non 0
        zero_placeholder = np.min([value for value in motion if value > 0])
        motion = [value if value > 0 else zero_placeholder for value in motion]
        return np.array(motion)

    def normalize_vs_by_motion(
        self,
        visceral_slide,
        contour,
        normalization_type,
        normalization_df,
        norm_vicinity,
    ):
        """
        Normalises visceral slide by motion according to the specified normalisation type
        Parameters
        ----------
        visceral_slide : ndarray
           The visceral slide to normalise
        contour : Contour
           The contour for which the visceral slide was calculated
        normalization_type : VSNormType
           The type of normalisation to apply
        normalization_df : ndarray
           The 2-channels deformation field to use for normalisation
        norm_vicinity : int
           The size of vicinity for vicinity normalisation

        Returns
        -------
        visceral_slide : ndarray
           A normalised visceral slide
        """
        if normalization_type == VSNormType.average_anterior_wall:
            x_aw, y_aw, u_aw, v_aw = get_anterior_wall_data(
                contour.x_neighbour,
                contour.y_neighbour,
                contour.u_normal,
                contour.v_normal,
            )
            abdominal_wall_motion = self.get_motion(
                normalization_df, x_aw, y_aw, u_aw, v_aw
            )
            visceral_slide = visceral_slide / np.mean(abdominal_wall_motion)
        elif normalization_type == VSNormType.contour_vicinity:
            contour_motion = self.get_motion(
                normalization_df,
                contour.x_neighbour,
                contour.y_neighbour,
                contour.u_normal,
                contour.v_normal,
            )
            contour_motion_averaged = average_by_vicinity(contour_motion, norm_vicinity)
            visceral_slide = visceral_slide / contour_motion_averaged

        return visceral_slide


class VisceralSlideDetectorDF(VisceralSlideDetector):
    def get_visceral_slide(
        self,
        df_cavity,
        df_rest,
        df_normalization,
        moving_mask,
        normalization_type=VSNormType.none,
        norm_vicinity=15,
    ):
        """
        Calculates visceral slide based on the passed deformation field and abdominal cavity mask
        and normalises it with specified normalisation option
        Parameters
        ----------
        df_cavity : ndarray
           The 2-channels masked deformation field of abdominal cavity content
        df_rest : ndarray
           The 2-channels masked deformation field of abdominal cavity surroundings
        df_normalization : ndarray
           The 2-channels deformation field to use for normalisation
        moving_mask : ndarray
           The abdominal cavity segmentation on a moving frame
        normalization_type : VSNormType, default = VSNormType.none
           The type of normalisation to apply
        norm_vicinity : int, default = 15
           The size of vicinity for vicinity normalisation

        Returns
        -------
        x, y, visceral_slide : ndarray
            Coordinates of moving_mask contour and absolute values of visceral slide at these coordinates
        """

        df_full = df_cavity + df_rest

        contour = Contour.from_mask(moving_mask)
        visceral_slide = self.calculate_visceral_slide(df_full, contour)
        visceral_slide = np.abs(visceral_slide)

        # Normalize with the provided option
        visceral_slide = self.normalize_vs_by_motion(
            visceral_slide, contour, normalization_type, df_normalization, norm_vicinity
        )

        return contour.x, contour.y, visceral_slide


class VisceralSlideDetectorReg(VisceralSlideDetector):
    def __init__(self):
        self.registrator = Registrator()

    def get_visceral_slide(
        self,
        moving,
        moving_mask,
        fixed,
        fixed_mask=None,
        normalization_type=VSNormType.none,
        normalization_field=VSNormField.rest,
        norm_vicinity=15,
    ):
        """Calculates visceral slide based on the moving and fixed frames and their masks using image registration
        and normalises it with specified normalisation option

        Parameters
        ----------
        moving : ndarray
           Frame of cine MRI series to use as a moving image
        moving_mask : binary ndarray
           A binary segmentation of the abdominal cavity on moving frame
        fixed : ndarray
           Frame of cine MRI series to use as a fixed image
        fixed_mask : binary ndarray, optional
           A binary segmentation of the abdominal cavity on fixed frame
        normalization_type : VSNormType, default = VSNormType.none
           A type of visceral slide normalization to apply
        normalization_field : VSNormField, default=VSNormField.rest
           Specifies which deformation filed to use for visceral slide normalization
        norm_vicinity : int, default = 15
           A vicinity to use for VSNormType.contour_vicinity normalization type

        Returns
        -------
        x, y, visceral_slide : ndarray
            Coordinates of moving_mask contour and absolute values of visceral slide at these coordinates
        """
        moving = moving.astype(np.uint32)
        fixed = fixed.astype(np.uint32)

        moving_mask = moving_mask.astype(np.uint8)
        # We need to calculate the transformation between entire moving and fixed frame
        # if the moving mask is missing or we want to normalise the VS by the entire deformation filed
        complete_df_needed = fixed_mask is None or (
            normalization_type != VSNormType.none
            and normalization_field == VSNormField.complete
        )

        if complete_df_needed:
            # Register moving to fixed without mask
            transforms, complete_df = self.registrator.register(fixed, moving)

        if fixed_mask is None:
            # If the mask of the fixed image is not supplied, compute it through deformation of moving mask
            # Propagate moving mask to fixed
            fixed_mask = ants.apply_transforms(
                fixed=numpy_2d_to_ants(fixed),
                moving=numpy_2d_to_ants(moving_mask),
                transformlist=transforms["fwdtransforms"],
            ).numpy()

        fixed_mask = fixed_mask.astype(np.uint8)

        # Compute full deformation field as a sum of abdominal cavity and surroundings deformations
        deformation_field = self.get_full_deformation_field(
            fixed, moving, fixed_mask, moving_mask
        )

        contour = Contour.from_mask(moving_mask)
        visceral_slide = self.calculate_visceral_slide(deformation_field, contour)
        visceral_slide = np.abs(visceral_slide)

        self.normalization_df = (
            self.rest_field if normalization_field == VSNormField.rest else complete_df
        )
        visceral_slide = self.normalize_vs_by_motion(
            visceral_slide,
            contour,
            normalization_type,
            self.normalization_df,
            norm_vicinity,
        )

        return contour.x, contour.y, visceral_slide

    def get_full_deformation_field(
        self, fixed, moving, fixed_cavity_mask, moving_cavity_mask
    ):
        """
        Calculates the full deformation field
        Parameters
        ----------
        fixed, moving : ndarray
           A fixed and moving cine-MRI frame
        fixed_cavity_mask, moving_cavity_mask : ndarray
           Abdominal cavity segmentation masks on fixed and moving frames

        Returns
        -------

        """
        # Get cavity deformation field
        self.cavity_field = self.registrator.get_masked_deformation_field(
            fixed, moving, fixed_cavity_mask, moving_cavity_mask
        )
        # Get rest deformation field
        self.rest_field = self.registrator.get_masked_deformation_field(
            fixed, moving, 1 - fixed_cavity_mask, 1 - moving_cavity_mask
        )

        # Combine deformation fields into one
        return self.cavity_field + self.rest_field


class CumulativeVisceralSlideDetector:
    @staticmethod
    def warp_visceral_slide(x, y, visceral_slide, deformation_field):
        """
        Warps visceral slide by deformation field
        Parameters
        ----------
        x, y : ndarray
           The coordinates of visceral slide
        visceral_slide : ndarray
           The values of visceral slide
        deformation_field : ndarray
           A deformation field to perform warping
        Returns
        -------
        x, y, visceral_slide : list
           Visceral slide warped by deformation field
        """
        xy_warped = []
        for (current_x, current_y) in zip(x, y):
            u, v = deformation_field[current_y, current_x]
            xy_warped.append([round(current_x + u), round(current_y + v)])

        xy_warped = np.array(xy_warped)
        visceral_slide_warped = np.column_stack((xy_warped, visceral_slide))

        # Warped contour might have duplicated points
        # Find these points and take an average as VS slide value
        xy_warped_unique = np.unique(xy_warped, axis=0)

        visceral_slide_warped_unique = []
        for coord in xy_warped_unique:
            # Find all point with the current coordinate in the warped VS
            vs_at_coord = np.array(
                [
                    vs
                    for vs in visceral_slide_warped
                    if vs[0] == coord[0] and vs[1] == coord[1]
                ]
            )
            avg_vs = np.mean(vs_at_coord[..., 2])
            visceral_slide_warped_unique.append(avg_vs)

        return (
            xy_warped_unique[:, 0],
            xy_warped_unique[:, 1],
            visceral_slide_warped_unique,
        )

    @staticmethod
    def add_visceral_slides(visceral_slide, visceral_slide_warped):
        """
        Adds two visceral slides. Possible coordinates mismatches are handled by choosing a
        coordinate closest by Euclidean distance
        Parameters
        ----------
        visceral_slide : tuple of list
           The coordinates and values of the first visceral slide
        visceral_slide_warped : tuple of list
           The coordinates and values of the second visceral slide warped to match the first one

        Returns
        -------
        visceral_slide : tuple of list
           The coordinates and values of the first visceral slide computed as addition of the first and the second one
        """
        x_target, y_target, vs_target = visceral_slide
        x_warped, y_warped, vs_warped = visceral_slide_warped

        # Since some coordinates might mismatch after deformation,
        # the closest point of the total VS is used to sum with the current VS value
        vs_added = []
        for (x, y, vs) in zip(x_target, y_target, vs_target):
            diff = np.sqrt((x - x_warped) ** 2 + (y - y_warped) ** 2)
            index = np.argmin(diff)
            vs_added.append(vs + vs_warped[index])

        return x_target, y_target, np.array(vs_added)

    def compute_cumulative_visceral_slide(
        self, visceral_slides, frames, warping_dfs, plot=False
    ):
        """
        Computes cumulative visceral slide by adding visceral slides between subsequent cine-MRI frames pairs
        and averaging it
        Parameters
        ----------
        visceral_slides : list of tuple
           An ordered list of visceral slides subsequent cine-MRI frames pairs
        frames : list of ndarray
           An ordered list of cine-MRI frames that correspond to visceral slides
        warping_dfs : list of ndarray
           An ordered list of deformation fields to use for warping the cumulative visceral slide
        plot : bool, default=False
           A boolean flag indicating whether to visualise computation by plotting a current visceral slide,
           a warped cumulative visceral slide and new cumulative visceral slide at each step
        Returns
        -------
        total_x, total_y, total_vs : ndarray
           Coordinates and values of cumulative visceral slide
        """

        total_x = total_y = None
        total_vs = None

        for i, (x, y, vs) in enumerate(visceral_slides):

            if plot:
                frame = frames[i]
                plot_vs_on_frame(frame, x, y, vs, "VS {}".format(i))

            # At the first step, the visceral slide between first two frames is total
            if total_vs is None:
                total_x, total_y = x, y
                total_vs = vs
            else:
                # We want to transform the cumulative visceral slide to the frame corresponding to the current
                # visceral slide. That is the the current moving frame, hence we need the previous transformation
                # and we need to take the contour of the current moving image as the fixed image.
                warping_df = warping_dfs[i - 1]
                (
                    total_x_warped,
                    total_y_warped,
                    total_vs_warped,
                ) = self.warp_visceral_slide(total_x, total_y, total_vs, warping_df)

                if plot:
                    plot_vs_on_frame(
                        frame,
                        total_x_warped,
                        total_y_warped,
                        total_vs_warped,
                        "Total VS warped {}".format(i - 1),
                    )

                total_x, total_y, total_vs = self.add_visceral_slides(
                    (x, y, vs), (total_x_warped, total_y_warped, total_vs_warped)
                )

                if plot:
                    plot_vs_on_frame(
                        frame, total_x, total_y, total_vs, "Total VS {}".format(i - 1)
                    )

        # Take the average of total visceral slide
        total_vs /= len(visceral_slides)
        return total_x, total_y, total_vs


class CumulativeVisceralSlideDetectorDF(CumulativeVisceralSlideDetector):
    def __init__(self):
        self.vs_detector = VisceralSlideDetectorDF()

    def get_visceral_slide(
        self,
        moving_masks,
        cavity_dfs,
        rest_dfs,
        warping_dfs,
        normalization_dfs,
        normalization_type=VSNormType.none,
        norm_vicinity=15,
        plot=False,
    ):
        """
        Computes cumulative visceral slide based on the specified deformation fields, moving masks
        and normalization parameters
        Parameters
        ----------
        moving_masks : list of ndarray
           An ordered list of abdominal cavity segmentation masks on moving frame
           for each frames pair of cine-MRI series
        cavity_dfs : list of ndarray
           An ordered list of masked abdominal cavity deformation fields
        rest_dfs : list of ndarray
           An ordered list of masked abdominal cavity surroundings deformation fields
        warping_dfs : list of ndarray
           An ordered list of deformation fields to use for warping of cumulative visceral slide during summation
        normalization_dfs : list of ndarray
           An ordered list of deformation fields to use for visceral slide normalization
        normalization_type : VSNormType, default=VSNormType.none
           A type of visceral slide normalization to apply
        norm_vicinity : int, default=15
           A vicinity to use for VSNormType.contour_vicinity normalization type
        plot : bool, default=False
           A boolean flag indicating whether to visualise computation by plotting a current visceral slide,
           a warped cumulative visceral slide and new cumulative visceral slide at each step

        Returns
        -------
        total_x, total_y, total_vs : ndarray
           Coordinates and values of cumulative visceral slide
        """

        visceral_slides = []

        for i in range(len(moving_masks)):
            moving_mask = moving_masks[i]
            cavity_df = cavity_dfs[i]
            rest_df = rest_dfs[i]
            normalization_df = normalization_dfs[i]
            x, y, visceral_slide = self.vs_detector.get_visceral_slide(
                cavity_df,
                rest_df,
                normalization_df,
                moving_mask,
                normalization_type,
                norm_vicinity,
            )

            visceral_slides.append((x, y, visceral_slide))

        total_x, total_y, total_vs = self.compute_cumulative_visceral_slide(
            visceral_slides, moving_masks, warping_dfs, plot
        )
        return total_x, total_y, total_vs


class CumulativeVisceralSlideDetectorReg(CumulativeVisceralSlideDetector):
    def __init__(self):
        self.vs_detector = VisceralSlideDetectorReg()
        self.registrator = Registrator()

    def get_visceral_slide(
        self,
        series,
        masks,
        warping_field=VSWarpingField.contours,
        normalization_type=VSNormType.none,
        normalization_field=VSNormField.rest,
        norm_vicinity=15,
        plot=False,
        vs_computation_input_path=None,
    ):
        """
        Computes the cumulative visceral slide across the series in a slice
        Total visceral slide is matched with the current vs by warping it with deformation field
        and finding the contour point which is the closest to the current one
        Parameters
        ----------
        series : ndarray
           A cine-MRI slice to compute the cumulative visceral slide
        masks : ndarray
           An abdominal cavity segmentation corresponding to the cine-MRI slice
        warping_field : VSWarpingField, default=VSWarpingField.contours
           Specifies which deformation filed to use for visceral slide warping during addition
        normalization_type : VSNormType, default = VSNormType.none
           A type of visceral slide normalization to apply
        normalization_field : VSNormField, default=VSNormField.rest
           Specifies which deformation filed to use for visceral slide normalization
        norm_vicinity : int
           A vicinity to use for VSNormType.contour_vicinity normalization type
        plot: bool
           A boolean flag indicating whether to visualise computation by plotting a current visceral slide,
           a warped cumulative visceral slide and new cumulative visceral slide at each step

        Returns
        -------
        total_x, total_y, total_vs : ndarray
           Coordinates and values of cumulative visceral slide
        """

        # First, compute and save visceral slide for each subsequent pair of frames and contours transformation
        visceral_slides = []
        moving_masks = []
        warping_fields = []
        cavity_dfs = []
        rest_dfs = []
        normalization_dfs = []
        for i in range(1, len(series)):
            print("Processing pair {}".format(i))
            # Taking previous frames as moving and the next one as fixed
            moving = series[i - 1].astype(np.uint32)
            moving_mask = masks[i - 1].astype(np.uint32)
            moving_masks.append(moving_mask.copy())

            fixed = series[i].astype(np.uint32)
            fixed_mask = masks[i].astype(np.uint32)

            # Get visceral slide
            x, y, visceral_slide = self.vs_detector.get_visceral_slide(
                moving,
                moving_mask,
                fixed,
                fixed_mask,
                normalization_type,
                normalization_field,
                norm_vicinity,
            )

            visceral_slides.append((x, y, visceral_slide))

            if warping_field == VSWarpingField.contours:
                _, deformation_field = self.__contour_transforms(
                    fixed_mask, moving_mask, np.iinfo(np.uint16).max
                )
                warping_fields.append(deformation_field)
            else:
                warping_fields.append(self.vs_detector.rest_field)

            cavity_dfs.append(self.vs_detector.cavity_field.copy())
            rest_dfs.append(self.vs_detector.rest_field.copy())
            normalization_dfs.append(self.vs_detector.normalization_df.copy())

        if vs_computation_input_path is not None:
            self._pickle_vs_computation_input(
                vs_computation_input_path,
                moving_masks,
                cavity_dfs,
                rest_dfs,
                warping_fields,
                normalization_dfs,
            )

        total_x, total_y, total_vs = self.compute_cumulative_visceral_slide(
            visceral_slides, series, warping_fields, plot
        )
        return total_x, total_y, total_vs

    def _pickle_vs_computation_input(
        self,
        filepath,
        moving_masks,
        cavity_dfs,
        rest_dfs,
        warping_dfs,
        normalization_dfs,
    ):
        vs_computation_input = {}
        vs_computation_input["moving_masks"] = moving_masks
        vs_computation_input["cavity_dfs"] = cavity_dfs
        vs_computation_input["rest_dfs"] = rest_dfs
        vs_computation_input["warping_dfs"] = warping_dfs
        vs_computation_input["normalization_dfs"] = normalization_dfs
        with open(filepath, "w+b") as pkl_file:
            pickle.dump(vs_computation_input, pkl_file)
        print(f"Pickled vs computation input to {filepath}")

    def __contour_transforms(self, fixed_mask, moving_mask, contour_value):
        """
        Computes transformation between contours of two masks by converting masks images to contour images
        and performing image registration
        Parameters
        ----------
        fixed_mask : ndarray
           An image of a fixed mask
        moving_mask : ndarray
           An image of a moving mask
        contour_value : int
            A value to fill in at the contour coordinates

        Returns
        -------
        transforms, deformation_field
           Transformations between moving and fixed contours computed by ANTS toolkit
        """
        fixed_contour = mask_to_contour(fixed_mask, contour_value)
        moving_contour = mask_to_contour(moving_mask, contour_value)

        return self.registrator.register(fixed_contour, moving_contour)
