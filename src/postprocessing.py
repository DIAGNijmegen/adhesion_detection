import SimpleITK as sitk
from scipy import ndimage
from pathlib import Path
from data_conversion import convert_2d_image_to_pseudo_3d


# fill in binary holes in nnUNet prediction
def fill_in_holes(masks_path):
    """
    Fills in binary holes in nnUNet prediction

    We know that abdominal cavity represents a single connected area without holes
    hence the prediction can be improved if we fill in the holes in a segmentation mask

    Parameters
    ----------
    masks_path : Path
       A path to a folder which contains masks predicted by nnU-Net
    """
    slices_metadata_glob = masks_path.glob("*.nii.gz")
    for mask_path in slices_metadata_glob:
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))[0]
        mask = ndimage.binary_fill_holes(mask)
        mask_pseudo_3d = convert_2d_image_to_pseudo_3d(mask, is_seg=True)
        sitk.WriteImage(mask_pseudo_3d, str(mask_path))
