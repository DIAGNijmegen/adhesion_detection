import numpy as np
import matplotlib.pyplot as plt
from vs_definitions import VSTransform
from contour import filter_out_prior_vs_subset


def plot_vs_distribution(visceral_slides,
                         output_path,
                         vs_min=-np.inf,
                         vs_max=np.inf,
                         transform=VSTransform.none,
                         prior_only=False):
    """
    Plots a histogram and a box plot of visceral slide values in the passed set, optionally with removed ourliers
    and applied transformation
    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       The set of visceral slide to plot the statistics for
    output_path : Path
       The path where to save the histogram and the box plot
    vs_min: float, default=-np.inf
       A lower limit of visceral slide to consider
    vs_max: float, default=np.inf
       An upper limit of visceral slide to consider
    transform : VSTransform, default=VSTransform.none
       A transformation to apply to visceral slide values
    prior_only : bool, default=False
       A boolean flag indicating whether to only include visceral slide values from the adhesion prior region
    """

    output_path.mkdir(exist_ok=True, parents=True)

    vs_values = []
    for visceral_slide in visceral_slides:
        # Leave out regions which cannot have adhesions if the option is specified
        if prior_only:
            prior_subset = filter_out_prior_vs_subset(visceral_slide)
            cur_vs_values = [vs for vs in prior_subset[:, 2] if vs_min <= vs <= vs_max]
            vs_values.extend(cur_vs_values)
        else:
            cur_vs_values = [vs for vs in visceral_slide.values if vs_min <= vs <= vs_max]
            vs_values.extend(cur_vs_values)

    if transform == VSTransform.log:
        vs_values = [vs for vs in vs_values if vs >0]
        vs_values = np.log(vs_values)
    elif transform == VSTransform.sqrt:
        vs_values = np.sqrt(vs_values)

    def title_suffix():
        suffix = "_prior" if prior_only else "_all"
        suffix += "_outl_removed" if vs_max < np.inf else ""
        if transform == VSTransform.log:
            suffix += "_log"
        elif transform == VSTransform.sqrt:
            suffix += "_sqrt"

        return suffix

    bp_tile = "vs_boxplot" + title_suffix()
    # Boxplot
    plt.figure()
    plt.boxplot(vs_values)
    plt.savefig(output_path / bp_tile, bbox_inches='tight', pad_inches=0)
    plt.show()

    hs_tile = "vs_hist" + title_suffix()
    # Histogram
    plt.figure()
    plt.hist(vs_values, bins=200)
    plt.savefig(output_path / hs_tile, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_exhaustive_vs_distribution(visceral_slides, vs_min, vs_max, output_path):
    """
    Makes and saves histograms and box plots of visceral slide distribution for all possible
    options of
    1) outlier removal by discarding values outside (vs_min, vs_max) range
    2) considering only visceral slide value from the adhesion prior region
    3) transformation options
    Parameters
    ----------
    visceral_slides : list of VisceralSlide
       The set of visceral slide to plot the statistics for
    vs_min: float
       A lower limit of visceral slide to consider
    vs_max: float
       An upper limit of visceral slide to consider
    output_path : Path
       The path where to save the histograms and the box plots
    """

    # Distribution for all
    plot_vs_distribution(visceral_slides, output_path)
    # Distribution for prior region
    plot_vs_distribution(visceral_slides, output_path, prior_only=True)
    # Distribution for all outliers removed
    plot_vs_distribution(visceral_slides, output_path, vs_min, vs_max)
    # Distribution for prior region outliers removed
    plot_vs_distribution(visceral_slides, output_path, vs_min, vs_max, prior_only=True)
    # Distribution for all sqrt transfrom
    plot_vs_distribution(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.sqrt)
    # Distribution for prior region sqrt transfrom
    plot_vs_distribution(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.sqrt, prior_only=True)
    # Distribution for all sqrt transfrom
    plot_vs_distribution(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.log)
    # Distribution for prior region sqrt transfrom
    plot_vs_distribution(visceral_slides, output_path, vs_min, vs_max, transform=VSTransform.log, prior_only=True)


def plot_visceral_slide_expectation(means, vs, title, frame, output_path):
    """
    Plots visceral slide expectation visualised on a visceral slide contour
    Parameters
    ----------
    means : list of float
       An array of visceral slide expectation by location at the contour
    vs : VisceralSlide
       A visceral slide which contour use for visualisation
    title :
       A title to use to save the plot
    frame :
       A cine-MRI frame on which to visualise the expectation. Should match the visceral slide contout
    output_path : Path
       A path where to save cine-MRI slide
    """
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
    plt.scatter(x, y, s=5, c=c, cmap="jet")
    plt.colorbar()
    x_bl, y_bl = vs.bottom_left_point
    plt.scatter(x_bl, y_bl, s=25, c="white")
    plt.axis("off")
    plt.savefig(output_path / "{}.png".format(title), bbox_inches='tight', pad_inches=0)
    plt.close()