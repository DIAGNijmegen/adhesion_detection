# Functions to handle the updated patients mapping, which specified a correct patient id for each old patient id
# extracted as patient_id_mapping.json
import json

def map_bb_annotations(annotations_path, mapping_path, mapped_file_path):
    """
    Fixes patients ids in a file with adhesion annotation with bounding box
    Parameters
    ----------
    annotations_path : Path
       A path to a metadata file with adhesion annotation with bounding box
    mapping_path : Path
       A path to a file with mapping of old patient ids to correct one
    mapped_file_path : Path
       A path where to save updated bounding box annotations

    Returns
    -------

    """
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
    """
    Check how many patients from the old dataset (ANON) has more than one studies
    Parameters
    ----------
    mapping_path : Path
       A path to a file with mapping of old patient ids to correct one
    """
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
            print("Patient {} has multiple mapping".format(mapped_key))


def rename_image_folders(mapping_path, images_path):
    """
    Renames patient folders according to the new mapping
    Parameters
    ----------
    mapping_path : Path
       A path to a file with mapping of old patient ids to correct one
    images_path : Path
       A path to a folder that contain patients folders
    """
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





