import sys
import argparse
import numpy as np
from pathlib import Path
from cinemri.utils import get_patients, Patient
from config import IMAGES_FOLDER
from data_conversion import extract_frames, merge_frames

SEPARATOR = "_"
FRAMES_FOLDER = "frames"
METADATA_FOLDER = "images_metadata"


# function to extract frames and metadata
def extract_segmentation_data(archive_path,
                              target_path,
                              target_frames_folder=FRAMES_FOLDER,
                              target_metadata_folder=METADATA_FOLDER,
                              images_folder=IMAGES_FOLDER):

    target_path.mkdir()

    target_images_path = target_path / target_frames_folder
    target_images_path.mkdir()

    target_metadata_path = target_path / target_metadata_folder
    target_metadata_path.mkdir()

    patients = get_patients(archive_path / images_folder)
    for patient in patients:
        for (scan_id, slices) in patient.scans.items():
            for slice in slices:
                slice_path = archive_path / images_folder / patient.id / scan_id / slice
                slice_id = SEPARATOR.join([patient.id, scan_id, slice[:-4]])
                extract_frames(slice_path, slice_id, target_images_path, target_metadata_path)


def extract_data(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('archive_path', type=str, help="a path to the full cine-MRI data archive")
    parser.add_argument('target_path', type=str, help="a path to save the extracted frames")
    args = parser.parse_args(argv)

    archive_path = Path(args.archive_path)
    target_path = Path(args.target_path)
    extract_segmentation_data(archive_path, target_path)


def merge_segmentation(segmentation_path,
                       metadata_path,
                       target_path):
    target_path.mkdir(exist_ok=True)

    # Get slices ids from metadata folder
    slices_metadata_glob = metadata_path.glob("*.json")
    slices_id_chunks = [f.name[:-5].split("_") for f in slices_metadata_glob]
    slices_id_chunks = np.array(slices_id_chunks)

    # Extract patients to recover original folder structure
    patient_ids = np.unique(slices_id_chunks[:, 0])
    patients = []
    for patient_id in patient_ids:
        patient = Patient(patient_id)
        slices = slices_id_chunks[slices_id_chunks[:, 0] == patient_id]
        for _, scan_id, slice_id in slices:
            patient.add_slice(slice_id, scan_id)

        patients.append(patient)

    for patient in patients:
        patient_path = target_path / patient.id
        patient_path.mkdir()
        for scan_id in patient.scan_ids:
            scan_path = patient_path / scan_id
            scan_path.mkdir()

            for slice_id in patient.scans[scan_id]:
                slice_name = SEPARATOR.join([patient.id, scan_id, slice_id])
                slice_metadata_path = metadata_path / (slice_name + ".json")

                # Extract frames matching the slice id and its metadata, merge into .mha
                # and save in a scan folder
                slice_glob_pattern = slice_name + "*.nii.gz"
                merge_frames(slice_glob_pattern, slice_id, segmentation_path, scan_path, slice_metadata_path)


def merge(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation_path', type=str, help="a path to the predicted segmentation")
    parser.add_argument('metadata_path', type=str, help="a path to the images metadata")
    parser.add_argument('target_path', type=str, help="a path to save the merged prediction")
    args = parser.parse_args(argv)

    segmentation_path = Path(args.segmentation_path)
    metadata_path = Path(args.metadata_path)
    target_path = Path(args.target_path)
    merge_segmentation(segmentation_path, metadata_path, target_path)


def test():
    archive_path = Path("../../data/cinemri_mha/rijnstate1")
    target_path_images = Path("../../data/cinemri_mha/full_segmentation")
    target_path_metadata = Path("../../data/cinemri_mha/images_metadata")
    segmentation_path = Path("../../data/cinemri_mha/segmentation")
    merged_segmentation_path = Path("../../data/cinemri_mha/merged_segmentation")

    #extract_data(archive_path, target_path_images, target_path_metadata)
    #simulate_segmentation(target_path_images, segmentation_path)
    merge_segmentation(segmentation_path, target_path_metadata, merged_segmentation_path)


if __name__ == '__main__':
    actions = {
        'extract_data': extract_data,
        'merge': merge
    }

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print('Usage: data_conversion ' + '/'.join(actions.keys()) + ' ...')
    else:
        action(sys.argv[2:])





