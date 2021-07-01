import json
from pathlib import Path
from cinemri.config import ARCHIVE_PATH
from config import METADATA_FOLDER, BB_ANNOTATIONS_FILE, PATIENTS_METADATA_FILE_NAME, PATIENTS_METADATA_FILE_OLD_NAME, \
    PATIENTS_MAPPING_FILE_NAME, BB_ANNOTATIONS_EXPANDED_FILE, ANNOTATIONS_TYPE_FILE, IMAGES_FOLDER


def map_bb_annotations(annotations_path, mapping_path, mapped_file_path):
    # Read json
    with open(mapping_path) as f:
        mapping = json.load(f)

    # Read annotations
    with open(annotations_path) as f:
        annotations = json.load(f)

    mapped_annotations = {}
    for patient_id, studies_dict in annotations.items():
        mapped_patient_id = mapping[patient_id]
        if patient_id not in mapped_annotations:
            mapped_annotations[mapped_patient_id] = studies_dict
        else:
            for study_id, slices_dict in studies_dict.items():
                mapped_annotations[mapped_patient_id][study_id] = slices_dict
                print("Patient {} has not a single mapping".format(mapped_patient_id))

    # Write new annotations from file
    with open(mapped_file_path, "w") as f:
        json.dump(mapped_annotations, f)


def investigate_anon_mapping(mapping_path):
    # Read json
    with open(mapping_path) as f:
        mapping = json.load(f)

    mapping_keys = [key for key in mapping.keys() if key.startswith("ANON")]
    inverse_dict = {}
    for key in mapping_keys:
        mapped_key = mapping[key]
        if mapped_key not in inverse_dict:
            inverse_dict[mapped_key] = [key]
        else:
            inverse_dict[mapped_key] += key
            print("Patient {} has not a single mapping".format(mapped_key))


def rename_image_folders(mapping_path, images_path):
    # Read json
    with open(mapping_path) as f:
        mapping = json.load(f)

    patient_ids = [f.name for f in images_path.iterdir() if f.is_dir()]
    for patient_id in patient_ids:
        mapped_id = mapping[patient_id]
        mapped_folder = images_path / mapped_id
        old_folder = images_path / patient_id
        old_folder.rename(mapped_folder)

    print("done")


if __name__ == '__main__':
    archive_path = ARCHIVE_PATH
    annotation_path = archive_path / METADATA_FOLDER / ANNOTATIONS_TYPE_FILE
    annotation_expanded_path = archive_path / METADATA_FOLDER / BB_ANNOTATIONS_EXPANDED_FILE
    mapping_path = archive_path / METADATA_FOLDER / PATIENTS_MAPPING_FILE_NAME
    mapped_file_path = archive_path / METADATA_FOLDER / "annotations_type_mapped.json"
    patients_metadata_file_path = archive_path / METADATA_FOLDER / PATIENTS_METADATA_FILE_NAME
    patients_metadata_old_file_path = archive_path / METADATA_FOLDER / PATIENTS_METADATA_FILE_OLD_NAME
    images_path = archive_path / "segmentation_paramedian"

    #map_bb_annotations(annotation_path, mapping_path, mapped_file_path)
    #investigate_anon_mapping(mapping_path)
    #rename_image_folders(mapping_path, images_path)

    positive_stat(annotation_expanded_path, patients_metadata_old_file_path, mapping_path)






