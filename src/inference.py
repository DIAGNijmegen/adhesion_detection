import sys
import subprocess
import SimpleITK
from pathlib import Path

from data_extraction import extract_frames, merge_frames
from visceral_slide import VSNormType, VSNormField, CumulativeVisceralSlideDetectorReg
from vis_visceral_slide import plot_vs_over_frame
from contour import filter_out_prior_vs_subset
from postprocessing import fill_in_holes

# files:
# data/image.mha - input slice (copy to container)
# data/mask.mha - mask obtained with nnU-Net (saved after inference)
# nnunet/input - nnU-Net input
# nnunet/output - nnU-Net prediction

data_dir = Path("data")

nnunet_input_dir = Path("nnunet/input")
nnunet_output_dir = Path("nnunet/output")

nnunet_model_dir = Path("nnunet/model")

FRAMES_FOLDER = "frames"
MASKS_FOLDER = "masks"
METADATA_FOLDER = "images_metadata"
MERGED_MASKS_FOLDER = "merged_masks"

SLICE_FILE = "image.mha"
SLICE_ID = "123"
VIS_FILE_NAME = "visceral_slide.png"


def compute_visceral_slide(image: SimpleITK.Image, nnUNet_model_path: Path, output_path: Path) -> SimpleITK.Image:
    subprocess.check_call(["ls", "-al"])

    # Make folder to extract frames of a cine-MRI slice and its metadata
    nnunet_input_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames and metadata for inference with nn-UNet
    extract_frames(image, SLICE_ID, nnunet_input_dir, nnunet_input_dir)

    # Inference with nnU-Net
    # Create output folder for nn-UNet inside the container
    nnunet_output_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        "nnunet", "predict", "Task101_AbdomenSegmentation",
        "--results", str(nnUNet_model_path),
        "--input", str(nnunet_input_dir),
        "--output", str(nnunet_output_dir),
        "--network", "2d"
    ]

    print("Cmd {}".format(cmd))

    subprocess.check_call(cmd)

    # Post-processing of nnU-Net prediction
    fill_in_holes(output_path)

    # Merge nnU-Net prediction into a .mha masks file and save in the /data folder
    mask = merge_frames(SLICE_ID, nnunet_output_dir, data_dir, nnunet_input_dir)

    # Compute Visceral Slide
    visceral_slide_detector = CumulativeVisceralSlideDetectorReg()
    x, y, values = visceral_slide_detector.get_visceral_slide(image,
                                                              mask,
                                                              normalization_type=VSNormType.average_anterior_wall,
                                                              normalization_field=VSNormField.complete)

    # Leave adhesion prior region only
    prior_subset = filter_out_prior_vs_subset(x, y, values)
    x, y, values = prior_subset[:, 0], prior_subset[:, 1], prior_subset[:, 2]

    # visualisation on a single frame
    output_path.mkdir(exist_ok=True)
    vis_path = output_path / VIS_FILE_NAME
    frame = image[-2]
    plot_vs_over_frame(x, y, values, frame, vis_path)

    # TODO change the returned image
    return image


def run(argv):
    image = SimpleITK.ReadImage(str(data_dir / SLICE_FILE))
    output_path = Path("/mnt/netcache/pelvis/projects/evgenia/test")

    compute_visceral_slide(image, nnunet_model_dir, output_path)


if __name__ == '__main__':

    # Very first argument determines action
    actions = {
        "run": run
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: registration ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])