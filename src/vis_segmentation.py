# Functions to visualise abdominal cavity segmentation
import numpy as np
import subprocess
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk
from config import SEPARATOR
from cinemri.utils import get_patients


def save_visualised_prediction(images_path, predictions_path, target_path, save_gif=True):
    """
    Visualises the predicted masks as overlays on the frames and saves as png files and a gif merged from png files
    For a single cine-MRI slice
    Parameters
    ----------
    images_path : Path
       A path to a folder that contains frames extracted from a cine-MRI slice
    predictions_path : Path
       A path to a folder that contains predicted segmentation masks
    target_path : Path
       A path to a folder to save visualized prediction
    save_gif : bool, default=True
       A boolean flag indicating whether visualized prediction should be merged into a gif and saved
    """

    target_path.mkdir(exist_ok=True)

    png_path = target_path / "pngs"
    png_path.mkdir(exist_ok=True)

    # get frames ids
    files = images_path.glob("*.nii.gz")
    frame_ids = sorted([file.name[:-12] for file in files], key=lambda file_id: int(file_id.split("_")[-1]))

    for frame_id in frame_ids:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(images_path / (frame_id + "_0000.nii.gz"))))[0]
        nnUNet_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(predictions_path / (frame_id + ".nii.gz"))))[0]

        plt.figure()
        plt.imshow(image, cmap="gray")
        masked = np.ma.masked_where(nnUNet_mask == 0, nnUNet_mask)
        plt.imshow(masked, cmap='autumn', alpha=0.2)
        plt.axis("off")
        overlayed_mask_file_path = png_path / (frame_id + ".png")
        plt.savefig(overlayed_mask_file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    if save_gif:
        command = [
            "convert",
            "-coalesce",
            "-delay",
            "20",
            "-loop",
            "0",
            str(png_path) + "/*png",
            str(target_path) + "/" + frame_id[:-3] + ".gif",
        ]
        subprocess.run(command)


def save_visualised_full_prediction(images_path, predictions_path, target_path, save_gif=True):
    """
    Visualises the predicted masks as overlays on the frames and saves as a png files and a gif merged from png files
    For a set of cine-MRI slices. Standard folders hierarchy patient -> study -> slice is kept
    Parameters
    ----------
    images_path : Path
       A path to a folder that contains cine-MRI slices
    predictions_path : Path
       A path to a folder that contains predicted segmentation masks
    target_path : Path
       A path to a folder to save visualized prediction
    save_gif : bool, default=True
       A boolean flag indicating whether visualized prediction should be merged into a gif and saved
    """

    target_path.mkdir(exist_ok=True)

    # get frames ids
    patients = get_patients(images_path)
    for patient in patients:
        patient.build_path(target_path).mkdir(exist_ok=True)

        for study in patient.studies:
            study.build_path(target_path).mkdir(exist_ok=True)

            for slice in study.slices:
                # extract cine-MRI slice
                slice_path = slice.build_path(images_path)
                slice_img = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_path)))
                # extract predicted mask

                mask_path = slice.build_path(predictions_path)
                if not mask_path.exists():
                    continue

                mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

                vis_path = slice.build_path(target_path, extension="")
                vis_path.mkdir(exist_ok=True)

                # Make a separate folder for .png files
                png_path = vis_path / "pngs"
                png_path.mkdir(exist_ok=True)

                for ind, frame in enumerate(slice_img):
                    frame_mask = mask[ind]

                    plt.figure()
                    plt.imshow(frame, cmap="gray")
                    masked = np.ma.masked_where(frame_mask == 0, frame_mask)
                    plt.imshow(masked, cmap='autumn', alpha=0.2)
                    plt.axis("off")
                    overlayed_mask_file_path = png_path / "{}_{}.png".format(slice.id, ind)
                    plt.savefig(overlayed_mask_file_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                if save_gif:
                    command = [
                        "convert",
                        "-coalesce",
                        "-delay",
                        "20",
                        "-loop",
                        "0",
                        str(png_path) + "/*png",
                        str(vis_path) + "/" + slice.id + ".gif",
                    ]
                    subprocess.run(command)


def extract_vis_gifs(source_path, target_path):
    """
    Extract gifs with visualised segmentation and saves on the target part
    with a full id of a cine-MRI slice used as a unique name
    Parameters
    ----------
    source_path : Path
       A path to full segmentation visualisation that contains pngs and a gif
    target_path : Path
       A path where to save extracted gifs
    """

    target_path.mkdir(exist_ok=True)

    patient_ids = [f.name for f in source_path.iterdir() if f.is_dir()]
    for patient_id in patient_ids:
        patient_path = source_path / patient_id
        studies = [f.name for f in patient_path.iterdir() if f.is_dir()]

        for study_id in studies:
            study_path = patient_path / study_id
            slices = [f.name for f in study_path.iterdir() if f.is_dir()]

            for slice_id in slices:
                gif_source_path = study_path / slice_id / (slice_id + ".gif")
                gif_target_path = target_path / (SEPARATOR.join((patient_id, study_id, slice_id)) + ".gif")

                shutil.copy(gif_source_path, gif_target_path)