
# How to run a pipeline to calculate visceral slide for a set of patients

The `visceral_slide_pipeline.py` contains the code to automate calculation of visceral slide. It is possible to run the whole pipeline with one command only as well as run each step separately, which is useful in case a particular step has failed. For the whole pipeling and for each step the corresponding function that take full set of needed input parameters are created as well as command line wrappers to run them on the cluster.   

The pipeline expects the following input parameters: a path to the whole cine-MRI archive, a path to the results of the nnU-Net training, an output path and the key specifying a subbset of the patient to extract. Currenlty it can be "train" or "test" subset based on the training/testing split used to train the nnU-Net.   

To run the pipeline on cluster use the command:
```
./c-submit --priority=high --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g \
           evgeniamartynova 9692 1 doduo1.umcn.nl/evgenia/nnunet \ 
           visceral_slide pipeline /mnt/netcache/pelvis/projects/janesmith/data/cinemri_mha/rijnstate \
           /mnt/netcache/pelvis/projects/janesmith/experiments/nnUNet_training/results \
           --output  /mnt/netcache/pelvis/projects/janesmith/experiments/visceral_slide \ 
           --mode train
```


The pipeline includes the following steps:

### Extraction of the inspiration and expiration frames

The inspiration and expiration frames are those frames of a slice that were determined as capturing the opposite phases of breathing cycle with `InspExpDetector` from `abdomenmrus-cinemri` repository. This information is stored in `metadata/inspexp.json` of the cine-MRI archive. Only slices for the specified subset of patients are considered. The train/test split used for nnU-Net training should be saved as `metadata/segm_train_test_split.json` in the cine-MRI archive. The extracted frames are saved in the specified output folder in the format expected by nnU-Net implementation with suffixes "_[insp/exp]_0000.nii.gz", which helps to differentiate inspiration and expiration frames. 

To run this step separately on cluster use the command:
```
./c-submit --priority=high --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g \
           evgeniamartynova 9692 1 doduo1.umcn.nl/evgenia/nnunet \ 
           visceral_slide extract_frames /mnt/netcache/pelvis/projects/janesmith/data/cinemri_mha/rijnstate \
           --output  /mnt/netcache/pelvis/projects/janesmith/experiments/visceral_slide_results \ 
           --mode train
```

Note that output folder of this step serves as an imput folder for the next 2 steps.

### Running an inference with nnU-Net for the extracted frames

To run this step separately on cluster use the command:
```
./c-submit --priority=high --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g \
           evgeniamartynova 9692 1 doduo1.umcn.nl/evgenia/nnunet \ 
           visceral_slide extract_frames /mnt/netcache/pelvis/projects/janesmith/experiments/visceral_slide_results \
           --results /mnt/netcache/pelvis/projects/janesmith/experiments/nnUNet_training/results \
           --task Task101_AbdomenSegmentation
```

### Running an algorithm to compute visceral slide for each slice and save the results

To run this step separately on cluster use the command:
```
./c-submit --priority=high --require-cpus=9 --require-mem=28g --gpu-count=1 --require-gpu-mem=11g \
           evgeniamartynova 9692 1 doduo1.umcn.nl/evgenia/nnunet \ 
           visceral_slide compute /mnt/netcache/pelvis/projects/janesmith/experiments/visceral_slide_results
```

## Default input and output folder structure

The first step of the pipeline requires the following structure of the archive and exactly the same names of files and folders: 

```
 cinemri_mha/rijnstate
 ├── images
     ├── PatientId
         ├── ScanId
             ├── Sclice1Id
             ├── Sclice2Id
             ├── ...
             ├── ScliceNId
 ├── metadata
     ├── inspexp.json
     ├── segm_train_test_split.json
```

The results are saved in the specified output folder in the follwing format:
```
 visceral_slide_results
 ├── nnUNet_input
     ├── PatientId_ScanId_Sclice1Id_exp_0000.nii.gz
     ├── PatientId_ScanId_Sclice1Id_insp_0000.nii.gz
     ├── PatientId_ScanId_Sclice2Id_exp_0000.nii.gz
     ├── PatientId_ScanId_Sclice2Id_insp_0000.nii.gz
     ├── ...
 ├── nnUNet_masks
     ├── PatientId_ScanId_Sclice1Id_exp.nii.gz
     ├── PatientId_ScanId_Sclice1Id_insp.nii.gz
     ├── PatientId_ScanId_Sclice2Id_exp.nii.gz
     ├── PatientId_ScanId_Sclice2Id_insp.nii.gz
     ├── ...
     ├── plans.pkl
     ├── postprocessing.json
├── visceral_slide
    ├── PatientId
         ├── ScanId
             ├── Sclice1Id
                 ├── visceral_slide_overlayed.png
                 ├── visceral_slide.png
                 ├── visceral_slide.pkl
             ├── ...
         ├── ...
    ├── ...
```

`nnUNet_input` is the folder in which inspiration and expiration frames extracted at the first step are saved.   
In `nnUNet_masks` segmentation resuts for these frames are saved.   
Finally, `visceral_slide` stores visceral slide for each slice in 3 formats:
- Scatter plot with varied colour intensity depending on the degree of visceral slide as an overlay on the inspiration frame
- Only scatter plot on the white background
- Raw coordinates of the abdominal cavity contour x, y and the computed visceral_slide values in pickle format

