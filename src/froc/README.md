# AbdomenMRUS-pca-surveillance
MSc Thesis of Joeran Bosma

## Usage
The training Environment requires a couple of things for proper usage: msk-tiger, Weights & Biases and git. 

- **ExperimentSettings**: this is part of the [msk-tiger](https://github.com/DIAGNijmegen/msk-tiger) library from DIAG and comes pre-installed with the diag Docker base image. Using this utility, a snapshot of the code is saved for each experiment, as well as all arguments used to define the run. Two things are needed to specify the correct folder: an experiment name and the experiment folder. Specify the experiment name using `.... train.py [ExperimentName]` and the experiment folder using `--exp_dir joeran/Experiments` or similar. 
- **Weights & Biases**: logging metrics to [wandb.ai](https://wandb.ai) requires two steps:
    * add `wandb` to the Docker file: `RUN pip3.8 install wandb`
    * log in when running an experiment, add to the scriptX.sh: `wandb login [api key from wandb.ai/settings]`. 
- **git**: at the start of each run, I automatically perform a `git pull` to keep the code on Chancey up-to-date with this repository. Doing this is handles by the startup script by adding `git pull` to it. 

For an example Docker file and scriptX.sh, see [Example Docker file and startup script](#example-docker-file-and-startup-script). 

After proper setup, models can be trained on SOL using the `./c-submit` command as follows:  
`./c-submit  --require-mem=20g --require-cpus=8 --gpu-count=1 --require-gpu-mem=8g --constraint=Pascal {user} {ticket id} 24 doduo1.umcn.nl/joeran/tensorflow2:latest AbdomenMRUS-pca-surveillance/Environment train.py [model name] --aug_preset BaselineV1 --train_preset SingleScanBaselineV3 --model_preset unet++_ag_3d BaselineV1 --data_preset SingleScanBaselineV6_192 --training_mode segmentation --image_labels lbl [...more arguments...] --fold 0 --resume`  

The `fold` parameter is zero-based and configures five-fold cross-validation by default. The flag `--resume` is required when continuing an existing run. Use `--overwrite` to replace an older experiment with the same name. See [Example commands](#example-commands) for more examples. 

## Programming environment
The programming environment consists of utility scripts organised in low-level, mid-level and high-level functions, and notebooks using these utilities. These notebooks are automatically converted to their .py and .html counterparts, for usage as script on SOL and easy previewing. 

**Low level**  
- `data_utils`: used to locate scans, annotations and zonal segmentations, read those files, and other low-level data functions. 
- `augmentations`: implements augmentations for 3D volumes and tf.data-compatible helper functions.  
- `visualisation`: plotting functions to show scans, predictions and labels. Also has functions to orchestrate previewing samples. Defines plotting of metrics and patient timeline. 
- `analysis_helper`: read predictions, collect experiment configurations, and other functions to help analysis.

**Mid level**  
- `preprocess_data`: read and preprocess scans with either upscaled ADC/DWI or dynamic resolution. 
- `data_reader`: orchestrates the tf.data pipeline from either TFRecords or Numpy files. Also provides an iterator to read and preprocess scans on-the-fly. 
- `train_helper`: collect target labels from dataset, load model checkpoint or transfuse model weights, and other helper function to assist model training. 
- `evaluation`: implements the train and validation metrics for a model during training

**High level**  
- `evaluation_utils`: orchestrate the evaluation of a model and implements test time augmentations.
- `visualisation_utils`: orchestrate preview of scans, annotations and augmentations, including reading and preprocessing. 
- `analysis_utils`: prepare model evaluation (load trained model, validation dataset, config), grab cross-validation subject lists. 

**Notebooks**  
- `Environment`: environment to train supervised 3D classification and segmentation models. 
- `Environment-SimCLR`: environment for representation learning using SimCLRv2. 

## FROC evaluation pipeline
_(written 18 May 2021)_  
The FROC evaluation pipeline is provided in [Environment/deploy_FROC.py](https://github.com/DIAGNijmegen/AbdomenMRUS-pca-surveillance/blob/main/Environment/deploy_FROC.py). The implementation is derived from Matin's pipeline, and differs in three ways:
 - A single lesion candidate is not allowed to match multiple ground truth lesions, as detailed in [this TRAC ticket](https://repos.diagnijmegen.nl/trac/ticket/9299#comment:52), and the two following replies
 - Allows a static threshold to preprocess the softmax (as used by Matin), a case-dependent dynamic threshold, a fast case-dependent dynamic threshold and Otsu's case-dependent threshold. Model performances with the different thresholds are evaluated [here](https://repos.diagnijmegen.nl/trac/ticket/9299#comment:56). The dynamic threshold is explained [here](https://repos.diagnijmegen.nl/trac/ticket/9299#comment:49). 
 - Uses multiprocessing, speeding up computation, at the cost of a bit more complexity.

**Usage**:  
For normal usage of the FROC evaluation pipeline, use the function `compute_FROC` with parameters `pre_threshold`, `all_softmax`, `all_labels` and `subject_list`. Please note this function is written for 3D binary FROC analysis. By default, this function removes all lesion candidates of less than 10 voxels. 

- `pre_threshold`: use one of `dynamic`, `dynamic-fast`, `otsu`, or a float to use the static threshold. (recommended: `dynamic-fast` during model training and `dynamic` for final evaluation)
- `all_softmax`: numpy array of all softmax volumes to evaluate. Provide an array of shape `(num_samples, D, H, W)`, where D, H, W are the depth, height and width. I have not checked this, but the order of the dimensions should not matter, as long as it is consistent with `all_labels`. 
- `all_labels`: numpy array of all ground truth labels. Provide an array of the same shape as `all_softmax`. Use `1` to encode ground truth lesion, and `0` to encode background (or use one-hot encoding, which is converted internally). 

**Additional settings**:  
For more control over the FROC evaluation pipeline, use:
- `min_dice`: defines the minimal required Dice score between a lesion candidate and ground truth lesion, to be counted as a true positive detection. 
- `min_voxels_detection`: defines the minimal size of lesion candidates, in voxels. All structures that are extracted from the softmax with less voxels are removed, and are not considered in the evaluation pipeline. 
- `dynamic_threshold_factor`: (only for `dynamic` and `dynamic-fast` threshold) controls the size of the lesion candidates, which are defined to extend from the lesions peak confidence to `peak/dynamic_threshold_factor`. See [this ticket](https://repos.diagnijmegen.nl/trac/ticket/9299#comment:49). 


## Datasets
The data pipeline consists of two parts: 1) reading from file, and 2) processing and augmenting the data. For supervised training, the full data pipeline described below is used (if enabled in the configuration). For Contrastive Learning, the samples are processed a bit differently. Currently, the samples 'leave' the data pipeline after the shuffling. Subsequently, the dynamic resolution and augmentations are handled by the CustomAugment class. This is something that could definitely be improved upon. 

### 1. Base dataset
The current data pipeline supports three ways to read from file: TFRecords, Numpy and from raw (.mhd/.raw and .nii.gz) files. Generating the base dataset from raw files involves resampling the input scans using SimpleITK, which is considerably slower than using the pre-resampled TFRecords or Numpy files. The raw format is especially useful for debugging, and its preprocessing parameters can directly be used to generate TFRecords. 

Note: all three use caching during training, making the overall performance difference of about 1.5 seconds per sample amount to ~1-1.5 hours for ~3000 samples. 

The base dataset should yield samples of dynamic or static resolution, with the associated structures given below. The dynamic resolution allows storing and processing the low-resolution ADC, DWI and label, while maintaining the high resolution of the T2W scan. In the current setup, the ADC, DWI and label are stored at their (3.6x2.0x2.0) spacing, while the T2W scan is saved at either (3.6x0.3x0.3) or (3.6x0.5x0.5) mm/voxel. This dynamic resolution is required to use the Rician noise augmentation, as that needs to be applied at the original scanning resolution. 

**Dynamic resolution**  
```
sample = {
    'features': {
        'img_T2W': 18xHxW scan, 
        'img_ADC': 18xhxw scan,
        'img_DWI': 18xhxw scan,
        'prior': 18xHxW map, # optional
        'zonal_mask': 18xHxW map, # optional
        'xy_dim_T2W': (H, W), # height and width of T2W scan (and prior and zonal_mask)
        'xy_dim': (h, w), # height and width of ADC and DWI scan
        ... # additional features (e.g. PSA density, prostate volume) go here
    },
    'labels': {
        'lbl': 18xhxw map, # optional
        ... # additinal labels (e.g. index lesion PI-RADS score, Gleason score) go here
    },
    'info': {
        ... # optional, additinal informating about the sample may go here
    }
}
```

**Static resolution**  
```
sample = {
    'features': {'x': 18xHxWxC scan},
    'labels': {'y': 18xHxW map}
}
```

### 2. Processing and augmenting the data
This part assumes samples of one of the two structures given above. The behaviour of the processing and augmentations are easiest to control using the Configurations class elaborated on below, but can also be used stand-alone. At the time of writing, the data pipeline consists of the following parts:

1. verify labels of samples, skips samples with nan values as label
2. filter excluded subjects: skips samples if their subject_id is in the exclusion list
3. cache dataset in RAM or to file
4. handle repetition and shuffling of the samples
5. for dynamic resolution: apply noisy instance-wise min-max normalisation, Rician noise, and resize scans to the target spacing (3.6x0.5x0.5)

At this point, both the originally dynamic resolution and originally static resolution samples have the structure, namely the stucture of static resolution samples. This is to facilitate a unified pipeline, and means the `sample['features']['x']` entry should be used to process the model input, instead of the individual entries for the different modalities. 

6. encode (segmentation) labels using one-hot or ordinal encoding
7. for granular segmentation maps, select the desired channels (e.g. only PI-RADS 2, 4 and 5)
8. augment the input scans and reflect e.g. rotation in the output labels
9. augment segmentation maps (e.g. label growing, small translations, Gaussian smoothing)
10. apply central crop, after augmentations, to reduce border interpolations
11. calculate derived labels (e.g. tumor volume change for pair-wise data pipeline)
12. encode feature scores with one-hot or ordinal encoding (e.g. prior Gleason score)
13. calculate active surveillance targets (should be merged with step k.)
14. impute missing feature scores 
15. apply data augmentation to tabular features
16. duplicate labels
17. prepare sample for keras (splits sample into features and labels, as required for Keras)
18. batch samples
19. apply MixUp and/or CutMix data augmentation, which require batches
20. prefetch for better computational performance 

## Datasets - usage
**From TFRecords**  
The dataset from TFRecords is defined by its `prep_dir`, `exclude_subject_ids` and `save_format=TFRecords`. The usage shoud be accompanied by the correct settings of the `patch_dims` (size before augmentations, e.g. `[18, 192, 192]`), `num_channels` (3 without prior, 4 with) and `num_classes` (2 for csPCa delineations). The model input size is subsequently controlled by the `central_crop_size` (size after augmentations, e.g. `144`). 
To ensure correct cross-validation splits and check if all files are found on Chancey, the number of train/validation parts needs to be specified with `num_trainval_parts`. By default, 15 train/validation parts are used, which enables easy 5-fold, 3-fold or 15-fold cross-validation with the same TFRecords. 

There exist multiple 'presets' of these settings, with three main categories:
1. SingleScanBaseline: dataset with a single visit per sample
2. Baseline: dataset with pairs of visits from the same patient per sample
3. SimCLRBaseline: dataset with single visits per sample, but excluding patients with multiple visits 

For Contrastive Learning, both 1. and 3. can be used. If using the contrastive learning to boost the performance of the Active Surveillance downstream task, it could be beneficial to use dataset 3. 

At the time of writing, the most recent version is V6. This version includes the prior, but does not employ the case-wise transformation. Also, this version does not use the prostate segmentation to center the samples. This version can be used by invoking:
`--data_preset SingleScanBaselineV6_192`.  

**From Numpy**  
The dataset from Numpy files is defined by its `prep_dir`, `csv_path_train`, `csv_path_val` and `save_format=Numpy`. This dataset should also be accompanied by the correct settings of the `patch_dims` (size before augmentations, e.g. `[18, 144, 1444]`), `num_channels` (3 without prior, 4 with) and `num_classes` (2 for csPCa delineations). The model input size is subsequently controlled by the `central_crop_size` (size after augmentations, e.g. `144`). Inclusion of the prior is handled by setting the `prior_path` to the correct location, and transformation to patient-wise anatomy is handled by the `transform_prior` flag. 

All file paths are relative to the project root, `~/pelvis/projects/`. For an example csv file, see e.g. `anindo/models/2020/medima2020/yann/feed/prostate-mpMRI_training-fold-1.csv`.  

**From raw**  
The dataset from raw format is defined by the settings specified in a .json file. To use this setup, set `save_format=raw` and specify the path to the configuration file with `ds_config_train` and `ds_config_val`. 

The configuration file should specify all options required to preprocess the raw files:
1. physical_size (in mm), e.g. (64.8, 96.0, 96.0) for 18x48x48 voxels at a spacing of 3.6x2.0x2.0, 18x192x192 voxels at 3.6x0.5x0.5 and 18x320x320 voxels at 3.6x0.3x0.3. 
2. in_dir_scans: list of parent folders where the .mhd/.raw files are located
3. in_dir_annot: list of parent folders where the .nii.gz files are located (optional)
4. in_dir_zonal: list of parent folders where the .nii.gz files are located (optional, required for transformation of prior and for centering the prostate)
5. prior_path: path to prior (as .npy file), if it should be included
6. metadata_fn: path to metadata Excel sheet (version 9 or later), if tabular features and labels should be included
7. metadata_float_features: which features to extract from the metadata
8. subject_list: list of subject ids to include

For an example, see e.g. `~/pelvis/projects/joeran/Data/prep/demo-cases-granular-pirads-val.json`. Please note that when specifying input folders, the subdirectories containing the word 'Detection' are automatically included. This means that `matin/Data/Annotations/` includes the contents of (at the time of writing) `Detection2016_my`, `from Detection2016`, `from Detection2017` and `from Detection2018`. However, adding `joeran/Data/Annotations/` does not include subdirectory `GranularPIRADS`. 


## Configurations
The settings for the training Environment are organised and defined by the Config classes from `configurations.py`. The settings are divided in five categories: 
1. Augmentation parameters
2. Training parameters
3. Model parameters
4. Data parameters
5. General parameters

The implementation of the configurations classes is optimized to be transparent about differences between presets. This does obscure a clear overview of the current settings from the code. To achieve this overview, it is best to define a set of configurations (either from code or by simulating a command in the training Environment) and use the `config.describe(full=True/False)` command. 

Configurations can be accessed directly by their case insensitive name, and for existing entries the 'subconfig' name can be ommitted. This means that, for example, the probability of applying CutMix can be accessed with `config.aug_config.cutmix_prob`, `config.aug_config.CutMix_prob`, `config.cutmix_prob` or similar. The entries can also be accessed like a dictionary, as e.g. `config['cutmix_prob']`. 

## Configurations - usage
Defining values for the configurations using the command line (for ./c-submit commands) is available for all entries listed in `configurations.py`. The specification of settings happens in two phases: 1) setting the presets, and 2) defining the modifications. These presets are available for the five configuration categories listed above, with a preset for each category individually. After setting these presets, further modifications are possible. Unlike the case insensitive access in code, the entries need to be specified case sensitive. Please refer to `configurations.py` for the exact spelling. 

Simple configurations are simple to set, e.g.:
```
--data_preset SingleScanBaselineV6_192
--max_epochs 100
--fold 0
```

Also configurations which are lists can be set, e.g.:
```
--simclr_g1_nodes 256 128
--image_features img_T2W img_ADC img_DWI
```

Empty lists are set with an additional space: e.g. `--image_labels  --next_entry`.  

For complex configurations the input from the command line is parsed as json string, which is currently the case for `pre_thresholds`, `strides`, `att_sub_samp` and `granular_segmentation_labels`. Setting these configurations should be possible as follows: 
```
--granular_segmentation_labels '[[2, "P2"], [5, "P5"]]'
```
However, on ./c-submit call the quotes are removed, rendering this approach ineffective. This is something that should be fixed still. 

For full examples, see the [Example commands below](#example-commands). 

## Example commands
Please note that the counting of folds is zero-based. 

**Train Contrastive Learning using TFRecords dataset**  
```
./c-submit  --require-mem=20g --require-cpus=8 --gpu-count=1 --require-gpu-mem=8g {user} 9299 120 doduo1.umcn.nl/joeran/tensorflow2:latest AbdomenMRUS-pca-surveillance/Environment train-SimCLR.py SimCLR_SingleScanBaselineV5_SEResNet_test_13 --aug_preset BaselineV1 --train_preset BaselineV0 --model_preset seresnet BaselineV0 --data_preset SingleScanBaselineV5_192 --num_channels 4 --training_mode simclr --clr_maxlr 0.045 --lr_mode CosineDecay --eLRdecay_steps 10000 --optimizer SGD --simclr_rot_degree 15.0 --simclr_shear 7.0 --simclr_trans_factor 0.10 --simclr_zoom 1.10 --simclr_g1_nodes 64 64 --simclr_noisy_minmax_sigma 0.05 --simclr_rician_noise_sigma 0.05 --central_crop_size 64 --batch_size 128 --max_epochs 500 --fold 0 --resume
```

**Train Contrastive Learning using Numpy dataset**  
```
./c-submit  --require-mem=20g --require-cpus=8 --gpu-count=1 --require-gpu-mem=8g --constraint=Pascal joeranbosma 9299 24 doduo1.umcn.nl/joeran/tensorflow2:latest AbdomenMRUS-pca-surveillance/Environment train-SimCLR.py SimCLR_yann_SEResNet_1 --aug_preset BaselineV1 --train_preset BaselineV0 --model_preset seresnet BaselineV0 --data_preset SingleScanBaselineV0 --prior_path 0 --num_channels 3 --training_mode simclr --csv_path_train anindo/models/2020/medima2020/yann/feed/prostate-mpMRI_training-fold-1.csv --csv_path_val anindo/models/2020/medima2020/yann/feed/prostate-mpMRI_validation-fold-1.csv --batch_size 13 --max_epochs 60 --clr_maxlr 0.003 --lr_mode CosineDecay --eLRdecay_steps 10000 --optimizer SGD --simclr_rot_degree 25.0 --simclr_shear 10.0 --simclr_trans_factor 0.20 --simclr_zoom 1.20 --simclr_g1_nodes 64 64 --simclr_noisy_minmax_sigma 0 --simclr_rician_noise_sigma 0
```

## Example Docker file and startup script
**Example Docker file**  
```
FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.4-pt1.7-v1

USER root

RUN pip3.8 install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN pip3.8 install autopep8
RUN pip3.8 install plotly
RUN pip3.8 install yapf
RUN pip3.8 install xlsxwriter
RUN pip3.8 install xlrd
RUN pip3.8 install openpyxl
RUN pip3.8 install tensorflow-addons
RUN pip3.8 install wandb

USER user

WORKDIR /mnt/netcache/pelvis/projects/joeran/

ENTRYPOINT /bin/bash Dockers/build_tf2_docker/scriptX.sh  $0 $@
```

**Example startup script**  
```
cd $1
echo Running in $1

LOG_DIR=/mnt/netcache/pelvis/projects/joeran/outputs/dashboard_logs
mkdir -p $LOG_DIR

LOG_PATH=$LOG_DIR/$SLURM_JOB_ID.log
echo Saving the log file in $LOG_PATH

nvidia-smi >> $LOG_PATH
env >> $LOG_PATH

# ls directories for higher chance of accurate filesystem
ls
ls Environment
ls Analysis
ls Preprocessing

# git pull
git status >> $LOG_PATH
git status
git pull

# Weights & Biases login
wandb login [API key from wandb.ai/settings]

echo "----------"
/usr/local/bin/python3.8 "${@:2}" --JOB-ID=$SLURM_JOB_ID | tee -a $LOG_PATH
```


**TODO: Gaussian label smoothing as label augmentation**  
**TODO: delete entries of individual modalities after they are merged into features->x.**  
**TODO: improve performance of Contrastive Learning pipeline by handling the duplication and augmentations within the main data pipeline.**  
**TODO: check if missing TFRecords actually result in the termination of the dataset setup.**

