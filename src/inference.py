#!/usr/local/bin/python3

import sys
import subprocess
import numpy as np
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
nnunet_model_dir = Path("nnunet/results")

SLICE_FILE = "image.mha"
SLICE_ID = "123"

VIS_FILE_NAME = "visceral_slide.png"
VIS_FILE_NAME_MHA = "visceral_slide.mha"
MASK_MHA = "mask.mha"


def compute_visceral_slide(input_image: SimpleITK.Image, output_path: Path) -> SimpleITK.Image:
    subprocess.check_call(["ls", "-al"])

    # Make folder to extract frames of a cine-MRI slice and its metadata
    nnunet_input_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames and metadata for inference with nn-UNet
    extract_frames(input_image, SLICE_ID, nnunet_input_dir, nnunet_input_dir)

    # Inference with nnU-Net
    # Create output folder for nn-UNet inside the container
    nnunet_output_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        "nnunet",
        "predict",
        "Task101_AbdomenSegmentation",
        "--results",
        str(nnunet_model_dir),
        "--input",
        str(nnunet_input_dir),
        "--output",
        str(nnunet_output_dir),
        "--network",
        "2d",
    ]

    print("Cmd {}".format(cmd))

    subprocess.check_call(cmd)

    subprocess.check_call(["ls", "nnunet/output", "-al"])

    # Post-processing of nnU-Net prediction
    fill_in_holes(nnunet_output_dir)

    # Merge nnU-Net prediction into a .mha masks file and save in the /data folder
    mask = merge_frames(SLICE_ID, nnunet_output_dir, data_dir, nnunet_input_dir)

    # Compute Visceral Slide
    input_image_np = SimpleITK.GetArrayFromImage(input_image)
    mask_np = SimpleITK.GetArrayFromImage(mask).astype(np.float32)
    visceral_slide_detector = CumulativeVisceralSlideDetectorReg()
    x, y, values = visceral_slide_detector.get_visceral_slide(
        input_image_np.astype(np.float32),
        mask_np,
        normalization_type=VSNormType.average_anterior_wall,
        normalization_field=VSNormField.complete,
    )

    # Leave adhesion prior region only
    prior_subset = filter_out_prior_vs_subset(x, y, values)
    x, y, values = prior_subset[:, 0], prior_subset[:, 1], prior_subset[:, 2]

    # visualisation on a single frame
    output_path.mkdir(exist_ok=True)
    vis_path = output_path / VIS_FILE_NAME
    frame = input_image_np[-2]
    plot_vs_over_frame(x, y, values, frame, vis_path)

    # Convert .png to .mha
    visceral_slide_mha = SimpleITK.ReadImage(str(vis_path))
    visceral_slide_mha_path = output_path / VIS_FILE_NAME_MHA
    SimpleITK.WriteImage(visceral_slide_mha, str(visceral_slide_mha_path))

    # Output mask
    mask_path = output_path / MASK_MHA
    SimpleITK.WriteImage(mask, str(mask_path))

    # TODO change the returned image
    return mask


def run(argv):
    image = SimpleITK.ReadImage(str(data_dir / SLICE_FILE))
    output_path = Path("/mnt/netcache/pelvis/projects/evgenia/test")

    compute_visceral_slide(image, output_path)


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