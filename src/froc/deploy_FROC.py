from __future__ import division
from __future__ import print_function
import argparse
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf
from scipy import ndimage
from skimage import measure, filters
from scipy.ndimage import gaussian_filter
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

'''
Binary PCa Detection in mpMRI
Script:         Dual-Stage FROC Analysis
Contributor:    anindox8, matinhz, joeranbosma
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         20/04/2021
'''


# Preproce=ss Softmax Volume (Clipping, Max Confidence)
def preprocess_softmax_static(softmax, threshold=0.10, min_voxels_detection=10, max_prob_round_decimals=4):
    """
    The minimum number of 10 voxels is currently not motivated. 
    Such a motivation could be based on the following assumptions:
    - NOT CHECKED: a csPCa lesion is at least 13 voxels on ADC/hbv (2x2x3.6 mm/voxel)
    - for a detection which has 100% overlap with such a lesion, the Dice Similarity
      Coefficient is DSC = 2*a / (n + a), where n is the number of voxels of the lesion 
      and a is the number of voxels of the detection. 
    
    Then, the minimum size of the detection is given by:
    a_min = n*DSC_min / (2 + DSC_min)

    For n=13 and DSC_min=0.1, this gives a_min = 10 voxels

    Note: in [Arif2020, https://doi.org/10.1007/s00330-020-07008-z] they
    specifically investigate low-risk patients. In that paper, the 
    smallest lesion considered is 0.03 cc. 
    At the resolution of T2 (0.5x0.5x3.6 mm/voxel), this corresponds
    to 33 voxels. 

    = Minimal Dice of 0.1
    Papers supporting a minimal Dice of 0.1:
    [1] "Optimizing size thresholds for detection of clinically significant prostate cancer on MRI: 
    Peripheral zone cancers are smaller and more predictable than transition zone tumors", 
    published in European Journal of Radiology: https://www.sciencedirect.com/science/article/pii/S0720048X20302606?via%3Dihub
    [2] McKinney, S.M., Sieniek, M., Godbole, V., Godwin, J., 2020. International Evaluation of an AI System 
    for Breast Cancer Screening. Nature 577, 89–94. doi:10.1038/s41586-019-1799-6.
    """
    # Load and Preprocess Softmax Image
    all_hard_blobs                                 = np.zeros_like(softmax)
    confidences                                    = []
    clipped_softmax                                = softmax.copy()
    clipped_softmax[softmax<threshold]             = 0
    blobs_index, num_blobs                         = ndimage.label(clipped_softmax, np.ones((3,3,3)))

    if num_blobs > 0:                              # For Each Prediction
        for tumor in range(1, num_blobs+1):
            # determine mask for current lesion
            hard_mask                              = np.zeros_like(blobs_index)
            hard_mask[blobs_index==tumor]          = 1

            if np.count_nonzero(hard_mask) <= min_voxels_detection:
                # remove tiny detection of <= 0.009 cm^3
                blobs_index[hard_mask.astype(bool)] = 0
                continue
            
            # add sufficiently sized detection
            hard_blob                              = hard_mask * clipped_softmax
            max_prob                               = np.max(hard_blob)
            if max_prob_round_decimals is not None:
                max_prob                           = np.round(max_prob, max_prob_round_decimals)
            hard_blob[hard_blob>0]                 = max_prob
            all_hard_blobs                        += hard_blob
            confidences.append((tumor,max_prob))
    return all_hard_blobs, confidences, blobs_index


def preprocess_softmax_dynamic(softmax, min_voxels_detection=10, num_lesions_to_extract=5,
                               dynamic_threshold_factor=2.5, max_prob_round_decimals=None,
                               remove_adjacent_lesion_candidates=True, verbose=False,
                               max_prob_failsafe_stopping_threshold=0.01):
    """
    Generate detection proposals using a dynamic threshold to determine the size of lesions. 
    Author: Joeran Bosma
    """
    working_softmax       = softmax.copy()
    dynamic_hard_blobs    = np.zeros_like(softmax)
    confidences           = []
    dynamic_indexed_blobs = np.zeros_like(softmax, dtype=int)

    #for tumor_index in range(1, num_lesions_to_extract+1): 
    while len(confidences) < num_lesions_to_extract:
        tumor_index = 1 + len(confidences)

        # determine max. softmax
        max_prob = np.max(working_softmax)

        if max_prob < max_prob_failsafe_stopping_threshold:
            break
        
        # set dynamic threshold to half the max
        threshold = max_prob / dynamic_threshold_factor
        
        # extract blobs for dynamix threshold
        all_hard_blobs, _, _ = preprocess_softmax_static(working_softmax, threshold=threshold, 
                                                         min_voxels_detection=min_voxels_detection,
                                                         max_prob_round_decimals=max_prob_round_decimals)
        
        # select blob with max. confidence
        # note: the max_prob is re-computed in the (unlikely) case that the max. prob
        # was inside a 'lesion candidate' of less than min_voxels_detection, which is
        # thus removed in preprocess_softmax_static. 
        max_prob = np.max(all_hard_blobs)
        mask_current_lesion = (all_hard_blobs == max_prob)
        
        # create mask with its confidence
        hard_blob = (all_hard_blobs * mask_current_lesion)

        # Detect whether the extractted mask is a ring/hollow sphere 
        # around an existing lesion candidate. For confident lesions, 
        # the surroundings of the prediction are still quite confident, 
        # and can become a second 'detection'. For an # example, please 
        # see extracted lesion candidates nr. 4 and 5 at:
        # https://repos.diagnijmegen.nl/trac/ticket/9299#comment:49
        # Detection method: grow currently extracted lesions by one voxel, 
        # and check if they overlap with the current extracted lesion.
        extracted_lesions_grown = ndimage.morphology.binary_dilation(dynamic_hard_blobs > 0)
        current_lesion_has_overlap = (mask_current_lesion & extracted_lesions_grown).any()

        # Check if lesion candidate should be retained
        if (not remove_adjacent_lesion_candidates) or (not current_lesion_has_overlap):
            # store extracted lesion
            dynamic_hard_blobs += hard_blob
            confidences        += [(tumor_index, max_prob)]
            dynamic_indexed_blobs += (mask_current_lesion * tumor_index)
        
        # remove extracted lesion from working-softmax
        working_softmax = (working_softmax * (~mask_current_lesion))
    
    return dynamic_hard_blobs, confidences, dynamic_indexed_blobs


def preprocess_softmax(softmax, threshold=0.10, min_voxels_detection=10, num_lesions_to_extract=5,
                       dynamic_threshold_factor=2.5, max_prob_round_decimals=None, remove_adjacent_lesion_candidates=True):
    if threshold == 'dynamic':
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_dynamic(softmax, min_voxels_detection=min_voxels_detection, 
                                                                               dynamic_threshold_factor=dynamic_threshold_factor,
                                                                               num_lesions_to_extract=num_lesions_to_extract,
                                                                               remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates)
    elif threshold == 'dynamic-fast':
        # determine max. softmax and set a per-case 'static' threshold based on that
        max_prob = np.max(softmax)
        threshold = max_prob / dynamic_threshold_factor
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_static(softmax, threshold=threshold, 
                                                                              min_voxels_detection=min_voxels_detection)
    elif threshold == 'otsu':
        threshold = filters.threshold_otsu(softmax)
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_static(softmax, threshold=threshold, 
                                                                              min_voxels_detection=min_voxels_detection)
    else:
        threshold = float(threshold) # convert threshold to float, if it wasn't already
        all_hard_blobs, confidences, indexed_pred = preprocess_softmax_static(softmax, threshold=threshold, 
                                                                              min_voxels_detection=min_voxels_detection)

    return all_hard_blobs, confidences, indexed_pred


# Compute Base Prediction Metrics (TP,FP,TN,FN)
def compute_pred_vector(softmax, label, min_dice=0.10, pre_threshold=0.10, min_voxels_detection=10,
                        dynamic_threshold_factor=2.5, dynamic_threshold_num_lesions=5,
                        remove_adjacent_lesion_candidates=True):
    y_list = []

    # Preprocess Softmax Volume
    _, confidences, indexed_pred = preprocess_softmax(softmax, threshold=pre_threshold, min_voxels_detection=min_voxels_detection,
                                                      dynamic_threshold_factor=dynamic_threshold_factor,
                                                      num_lesions_to_extract=dynamic_threshold_num_lesions,
                                                      remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates)

    if label.any():                         # For Each Malignant Scan
        labeled_gt, num_gt_lesions          = ndimage.label(label,  np.ones((3,3,3)))

        # For Each Tumor/Lesion in Ground-Truth Label
        for lesiong_id in range(1,num_gt_lesions+1):
            gt_lesion_mask                  = (labeled_gt == lesiong_id)

            # Index of Predicted Lesion Overlapping with Current GT Lesion
            overlapped_lesions_pred = list(np.unique(indexed_pred[gt_lesion_mask]))
            if 0 in overlapped_lesions_pred: overlapped_lesions_pred.remove(0)

            # Store Prediction Confidence for Current GT Lesion
            y_pred_for_target_gt = []    
            for lesion_id_pred, lesion_confidence in confidences:
                if lesion_id_pred in overlapped_lesions_pred:
                    y_pred_for_target_gt.append((lesion_id_pred, lesion_confidence))

            # No Prediction. Add FN.
            if   (len(y_pred_for_target_gt)==0):
                y_list.append((1,0))
            
            # Single Prediction. Add TP/FN. 
            elif (len(y_pred_for_target_gt)==1):
                lesion_id_pred    = y_pred_for_target_gt[0][0]
                lesion_confidence = y_pred_for_target_gt[0][1]
                lesion_pred_mask  = (indexed_pred == lesion_id_pred)
                dice_score        = dice_3d(lesion_pred_mask, gt_lesion_mask)
                
                # Match DSC to Assign TP/FN. Remove Prediction After Assignment.
                if dice_score > min_dice:
                    indexed_pred[lesion_pred_mask] = 0
                    y_list.append((1,lesion_confidence))  # Add TP

                else: y_list.append((1,0))                # Add FN
            
            # Multiple Predictions
            else:
                # Sort List based on Confidences
                y_pred_for_target_gt = sorted(y_pred_for_target_gt, key=takeSecond, reverse=True)

                gt_lesion_matched = False
                for lesion_id_pred, lesion_confidence in y_pred_for_target_gt:
                    lesion_pred_mask = (indexed_pred==lesion_id_pred)
                    dice_score       = dice_3d(lesion_pred_mask, gt_lesion_mask)

                    if dice_score > min_dice:
                        indexed_pred[lesion_pred_mask] = 0
                        y_list.append((1,lesion_confidence))
                        gt_lesion_matched = True
                        break
                
                if not gt_lesion_matched:
                    y_list.append((1,0))                # Add FN

        # Remaining Tumors/Lesions are FPs
        remaining_lesions = list(np.unique(indexed_pred))
        if 0 in remaining_lesions: remaining_lesions.remove(0)
        for lesion_id_pred, lesion_confidence in confidences:
            if lesion_id_pred in remaining_lesions:
                y_list.append((0,lesion_confidence))

    else: # For Benign Scan, All Predictions Are FPs
        num_gt_lesions = 0
        if len(confidences)>0:
            for _, lesion_confidence in confidences:
                y_list.append((0, lesion_confidence))
        else:   y_list.append((0,0)) # Avoid Empty List

    return y_list, num_gt_lesions


# Calculate FROC Metrics (FP Rate, Sensitivity)
def y_to_FROC(y_list_all, y_list_pnp,  total_patients,    total_normal_patients,
              threshold_mode='unique', num_thresholds=50, single_threshold=None):

    # Sort Predictions
    y_list_all.sort()
    y_list_pnp.sort()
    y_true_all          = []
    y_pred_all          = []
    y_pred_pnp          = []
    FP_per_image        = []
    FP_per_normal_image = []
    sensitivity         = []

    for tr, pr in y_list_all:
        y_true_all.append(tr)
        y_pred_all.append(pr)
    for _, pr in y_list_pnp: 
        y_pred_pnp.append(pr)

    # Total Number of Lesions
    y_true_all    = np.array(y_true_all)
    y_pred_all    = np.array(y_pred_all)
    y_pred_pnp    = np.array(y_pred_pnp)
    total_lesions = y_true_all.sum()

    # Compute Thresholds for FROC Analysis
    if (threshold_mode=='unique'):
        thresholds      = np.unique(y_pred_all)
        thresholds.sort()
        thresholds      = thresholds[::-1]
        low_thresholds  = thresholds[0:20]

        # For >1000 Thresholds: Resample to 1000 Keeping All Thresholds Higher Than 0.8 
        if (len(thresholds)>1000):
            rng         = np.arange(1,len(thresholds),len(thresholds)/1000, dtype=np.int32)
            st          = [thresholds[i] for i in rng]
            thresholds  = [t for t in thresholds if t > 0.8 or t in st or t in low_thresholds]
    elif (threshold_mode=='default'):
        thresholds      = np.linspace(0.0, 1.0, num_thresholds).tolist()
    elif (threshold_mode=='single'):
        thresholds      = [single_threshold]

    # For Each Threshold: Count FPs and Sensitivity
    for th in thresholds:
        if th>0:
            y_pred_all_thresholded                  = np.zeros_like(y_pred_all)
            y_pred_all_thresholded[y_pred_all > th] = 1
            y_pred_pnp_thresholded                  = np.zeros_like(y_pred_pnp)
            y_pred_pnp_thresholded[y_pred_pnp > th] = 1
            tp     = np.sum(y_true_all*y_pred_all_thresholded)
            fp     = np.sum(y_pred_all_thresholded - y_true_all*y_pred_all_thresholded)
            fp_pnp = np.sum(y_pred_pnp_thresholded)

            # Update FROC wth New Point
            FP_per_image.append(fp/total_patients)
            FP_per_normal_image.append(fp_pnp/total_normal_patients)
            sensitivity.append(tp/total_lesions)

        else:
            # Extend FROC curve to Infinity
            if (len(sensitivity)>0):
                sensitivity.append(sensitivity[-1])
                FP_per_image.append(10)
                FP_per_normal_image.append(10)

    return FP_per_image, FP_per_normal_image, sensitivity, thresholds



# Compute Full FROC 
def compute_FROC(softmax_dir=None, patch_dims=None, val_size=None, threshold_mode='unique', num_thresholds=50, 
                 pre_threshold=0.10, min_voxels_detection=10, dynamic_threshold_factor=2.5, dynamic_threshold_num_lesions=5, 
                 remove_adjacent_lesion_candidates=True, all_softmax=None, all_labels=None, subject_list=None, 
                 flat=None, num_parallel_calls=4):
    

    if all_softmax is None or all_labels is None:
        # Compile All Volumes in Single Sweep + Compute List of Thresholds
        counter     = 0
        all_softmax = np.zeros(shape=(val_size,patch_dims[0],patch_dims[1],patch_dims[2]), dtype=np.float64)
        all_labels  = np.zeros(shape=(val_size,patch_dims[0],patch_dims[1],patch_dims[2]), dtype=np.uint8)
        for f in os.listdir(softmax_dir):
            if '_softmax.npy' in f:
                all_softmax[counter,:,:,:] = np.load(softmax_dir+f)
                all_labels[counter,:,:,:]  = np.load(softmax_dir+f.split('_softmax')[0]+'_label.npy')
                counter += 1
    else:
        val_size = all_softmax.shape[0]

    if all_labels.shape[-1] == 2:
        # convert one-hot encoded label to single channel
        all_labels = all_labels[..., 1]
    if all_softmax.shape[-1] == 2:
        # convert softmax to single channel
        all_softmax = all_softmax[..., 1]

    # Initialize Lists
    roc_true              = {}
    roc_pred              = {}
    y_list_all            = []
    y_list_pnp            = []
    total_lesions         = 0

    if subject_list is None:
        subject_list = range(len(all_softmax))
        if flat is None: flat = True


    with ThreadPoolExecutor(max_workers = num_parallel_calls) as pool:
        # define the functions that need to be processed: compute_pred_vector, with each individual
        # softmax prediction, ground truth label and parameters
        future_to_args = {
            pool.submit(compute_pred_vector, y_pred, y_true, pre_threshold=pre_threshold, 
                        min_voxels_detection=min_voxels_detection,
                        dynamic_threshold_factor=dynamic_threshold_factor,
                        dynamic_threshold_num_lesions=dynamic_threshold_num_lesions,
                        remove_adjacent_lesion_candidates=remove_adjacent_lesion_candidates): idx 
            for (y_pred, y_true, idx) in zip(all_softmax, all_labels, subject_list)
        }
        
        # process the cases in parallel
        for future in concurrent.futures.as_completed(future_to_args):
            idx = future_to_args[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"Exception: {e}")
            else:
                # unpack results
                y_list_pat, num_lesions_gt = res
                
                # aggregate results
                roc_true[idx] = np.max([a[0] for a in y_list_pat])
                roc_pred[idx] = np.max([a[1] for a in y_list_pat])

                # Accumulate Outputs
                y_list_all        +=    y_list_pat
                if num_lesions_gt == 0:
                    y_list_pnp += y_list_pat
                total_lesions     +=    num_lesions_gt


    # Calculate statistics
    total_patients        = val_size
    total_normal_patients = sum([lbl == 0 for lbl in roc_true.values()])

    # Get Lesion-Based Results
    FP_per_image, FP_per_normal_image, sensitivity, thresholds = y_to_FROC(y_list_all, y_list_pnp, total_patients, total_normal_patients,
                                                                           threshold_mode=threshold_mode, num_thresholds=num_thresholds)
    
    if flat:
        # flatten roc_true and roc_pred
        roc_true = [roc_true[s] for s in subject_list]
        roc_pred = [roc_pred[s] for s in subject_list]

    return FP_per_image, FP_per_normal_image, sensitivity, thresholds, total_lesions, total_normal_patients, roc_true, roc_pred


# Dice Coefficient for 3D Volumes
def dice_3d(predictions, labels):
    epsilon     =  1e-7
    dice_num    =  np.sum(predictions[labels==1])*2.0 
    dice_denom  =  np.sum(predictions) + np.sum(labels)
    return ((dice_num+epsilon)/(dice_denom+epsilon)).astype(np.float32)

"""
# Generate Whole_Volume Softmax Predictions for FROC Analysis [Stage 1]
def softmax_for_FROC(model_path, csv_path, save_path, whole_image_shape, spatial_dim, 
                     prior_path=None, transform_prior=False, TTA=False, display=False):

    # Read CSV with Validation/Test Set
    file_names = pd.read_csv(csv_path, dtype=object, keep_default_na=False, na_values=[]).values

    # Load Trained Model
    export_dir = \
        [os.path.join(model_path, o) for o in sorted(os.listdir(model_path))
         if os.path.isdir(os.path.join(model_path, o)) and o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    predictor  = tf.contrib.predictor.from_saved_model(export_dir=export_dir)

    # For Decomposing 2D Slice-Wise Predictions to Full Volumes
    img_2d     = np.zeros(shape=whole_image_shape, dtype=np.float64)
    lbl_2d     = np.zeros(shape=whole_image_shape, dtype=np.int32)
    count_2d   = 0

    print('Test-Time Augmentation:', str(TTA))
    print('Spatial Dimensions:', str(spatial_dim))

    # Iterate through Files, Predict on the Full Volumes
    for output in detection_read_fn(file_references = file_names,
                                    mode            = tf.estimator.ModeKeys.EVAL,
                                    params          = {'prior':           prior_path,
                                                       'transform_prior': transform_prior,
                                                       'spatial_dim':     spatial_dim,
                                                       'display':         display      }):
        # Parse Data Reader Output
        img          = output['features']['x']
        lbl          = output['labels']['y']
        subject_id   = output['img_id']

        if TTA:
            # Test-Time Augmentations   
            base_img = img.copy()
            flip_img = np.flip(base_img, axis=2)

            # Generate Predictions (Batches of 2 to Alleviate Memory Constraints)
            y_prob           =  predictor.session.run(
                fetches      =  predictor._fetch_tensors['y_prob'],
                feed_dict    = {predictor._feed_tensors['x']: np.concatenate((np.expand_dims((base_img),axis=0),
                                                                              np.expand_dims((flip_img),axis=0)), axis=0)})
            # Reverse Augmentations and Aggregate Softmax TTA Predictions
            rev_flip_img = np.flip(y_prob[1], axis=2)
            y_prob       = np.concatenate((np.expand_dims((y_prob[0]),  axis=0),
                                           np.expand_dims((rev_flip_img), axis=0)), axis=0) 
        else:    
            # For Dual-GPU Training/Inference
            splitGPU_img = np.concatenate((np.expand_dims((img),axis=0),
                                           np.expand_dims((img),axis=0)), axis=0)  
            # Generate Predictions
            y_prob           =  predictor.session.run(
                fetches      =  predictor._fetch_tensors['y_prob'],
                feed_dict    = {predictor._feed_tensors['x']: splitGPU_img})

        # Export Predictions
        if (spatial_dim=='3D'):
            # Export Predictions and Corresponding Labels
            np.save(save_path+subject_id+'_softmax.npy', np.mean(y_prob, axis=0)[:,:,:,1])
            np.save(save_path+subject_id+'_label.npy',   lbl)
        elif (spatial_dim=='2D'):
            case_number                                          = count_2d//whole_image_shape[0]
            img_2d[count_2d-(whole_image_shape[0]*case_number)]  = y_prob[0][:,:,1]
            lbl_2d[count_2d-(whole_image_shape[0]*case_number)]  = lbl
            count_2d                                            += 1
            
            # Upon Populating Each 3D Volume with 2D Predictions
            if ((count_2d+1)%whole_image_shape[0]==0):

                # Export Predictions and Corresponding Labels
                np.save(save_path+subject_id+'_softmax.npy', img_2d)
                np.save(save_path+subject_id+'_label.npy',   lbl_2d)

                # Reset 3D Volume Placeholder
                img_2d     = np.zeros(shape=whole_image_shape)
                lbl_2d     = np.zeros(shape=whole_image_shape)
"""

"""
# Inference via Classifiers [Stage 2]
def all_classifiers_inference(model_path, csv_path, save_path, num_patches, 
                              min_tumor_vox, TTA_mode=False, display=False):
    
    assert (len(model_path)==len(save_path)), \
        'ERROR: Please Provide Model and Save Paths for All Classifier(s).'

    # For Each Classifier
    for i in range(len(model_path)):
        subject_id_list   =  []     # Patient ID
        label_list        =  []     # Labels
        class0_prob_list  =  []     # Class 0 Probability (Benign)
        class1_prob_list  =  []     # Class 1 Probability (Malignant)
        prediction_list   =  []     # Prediction

        # Read CSV with Validation/Test Set
        file_names = pd.read_csv(csv_path, dtype=object, keep_default_na=False, na_values=[]).values

        # Load Trained Models
        export_dir = \
            [os.path.join(model_path[i], o) for o in sorted(os.listdir(model_path[i]))
             if os.path.isdir(os.path.join(model_path[i], o)) and o.isdigit()][-1]
        print('Loading from {}'.format(export_dir)) 

        predictor = tf.contrib.predictor.from_saved_model(export_dir=export_dir)

        # Iterate through Files, Predict on the Full Volumes
        for output in classification_read_fn(file_references = file_names,
                                             mode            = tf.estimator.ModeKeys.EVAL,
                                             deploy_mode     = True,
                                             params          = {'extract_patches': False,
                                                                'n_patches':       num_patches,
                                                                'display':         False}):
            t0 = time.time() 

            # Parse Data Reader Output
            subject_id          = output['img_id']
            stage2_patch        = output['features']['x']
            stage2_lbl          = output['labels']['y']          

            for e in range(num_patches):
                # Test-Time Augmentation
                patch_img         = stage2_patch[e]
                if (TTA_mode==True):
                    flipped_patch = np.flip((patch_img), axis=1)
                    shifted_patch = ndimage.shift((patch_img),  shift=(0,3,3,0), mode='mirror')
                    rotated_patch = ndimage.rotate((patch_img), angle=2.5, axes=(1,2), reshape=False, mode='mirror')
                    scaled_patch  = ndimage.zoom((patch_img[:, patch_img.shape[1] // 16: - patch_img.shape[1] // 16,
                                                               patch_img.shape[2] // 16: - patch_img.shape[2] // 16, :]), zoom=(1,64/56,64/56,1))
                    all_patches   = np.vstack((np.expand_dims((patch_img),     axis=0), 
                                               np.expand_dims((flipped_patch), axis=0), 
                                               np.expand_dims((shifted_patch), axis=0), 
                                               np.expand_dims((rotated_patch), axis=0), 
                                               np.expand_dims((scaled_patch),  axis=0)))
                else:
                    all_patches   = np.expand_dims((patch_img), axis=0)

                # Classifier Inference
                y_                =  predictor.session.run(
                       fetches    =  predictor._fetch_tensors['y_prob'],
                       feed_dict  = {predictor._feed_tensors['x']: all_patches})

                # Assign Patch-Wise Label based on Threshold for Tumor Voxel Count
                if (np.count_nonzero(stage2_lbl[e]==3)>=min_tumor_vox[i]): stage2_patch_lbl = 1 
                else:                                                      stage2_patch_lbl = 0

                # Populate Lists
                subject_id_list.append(subject_id)
                label_list.append(stage2_patch_lbl)
                class0_prob_list.append(np.mean(np.array(y_)[:,0]))
                class1_prob_list.append(np.mean(np.array(y_)[:,1]))
                prediction_list.append(np.argmax(np.mean(np.array(y_), axis=0)))

                # Print Outputs
                if display: print('ID={}; Classifier 01: Prediction={}; True={}; Run Time={:0.2f} s;'.format(
                    subject_id, np.argmax(np.mean(np.array(y_), axis=0)), stage2_patch_lbl, time.time()-t0))

        # Export Stage 2 Classifier Predictions
        predictions = pd.DataFrame(list(zip(subject_id_list, label_list, class0_prob_list, class1_prob_list, prediction_list)), 
                                  columns=['subject_id',    'y_true',   'class0',         'class1',         'y_pred'])
        predictions.to_csv(save_path[i], encoding='utf-8', index=False)

"""


# Decision Fusion of Stage 2 Classifier Predictions with Stage 1 Softmax Predictions
def decision_fusion(lambda_penalty, stage2_case_threshold, stage2_patch_threshold, stage2_csv_path, stage1_save_path, 
                    dual_stage_save_path, stage2_roi_dims, stage2_num_patches, prior_path=None, display=False):
    
    num_classifiers     = len(stage2_csv_path)
    prediction_csv      = []
    predictions         = []    
    stage2_class0_probs = []
    stage2_class1_probs = []
    patch_counter       = 1  

    # Read CSV with Validation/Test Set Predictions from All Classifiers
    for i in range(num_classifiers): 
        prediction_csv.append(pd.read_csv(stage2_csv_path[i], dtype=object, keep_default_na=False, na_values=[]).values)

    # Iterate through Each Case
    for scan in range(len(prediction_csv[0])):
        '''
        Decision Fusion: Extract Overlapping ROI
        
        For a given softmax prediction volume from the first-stage detection network [18,144,144], we focus on a central crop covering
        the second-stage classifiers' total ROI [12,112,112], which is localized more tightly around the prostate. We decompose this ROI
        into 8 octant patches, matching the same spatial dimensions and orientation as those used to train the classifiers. 
        '''
        if ((patch_counter-1)%stage2_num_patches==0):
            subject_id                 = prediction_csv[0][scan][0]
            stage1_softmax             = np.load(stage1_save_path+subject_id+'_softmax.npy')
            stage1_lbl                 = np.load(stage1_save_path+subject_id+'_label.npy')            
            skip_slices_z              = (stage1_softmax.shape[0]-stage2_roi_dims[0])//2
            skip_slices_x              = (stage1_softmax.shape[1]-stage2_roi_dims[1])//2
            skip_slices_y              = (stage1_softmax.shape[2]-stage2_roi_dims[2])//2
            stage2_roi                 =  stage1_softmax[skip_slices_z      : skip_slices_z + stage2_roi_dims[0],
                                                         skip_slices_x      : skip_slices_x + stage2_roi_dims[1],
                                                         skip_slices_y      : skip_slices_y + stage2_roi_dims[2]].copy()

            # Load Prostate Cancer Probability Density Map (Prior) as Spatial Priori
            if (prior_path!=None):  prior = np.load(prior_path) 
            else:                   prior = np.zeros_like(stage1_softmax) 
            prior_roi             = prior[skip_slices_z : skip_slices_z + stage2_roi_dims[0],
                                          skip_slices_x : skip_slices_x + stage2_roi_dims[1],
                                          skip_slices_y : skip_slices_y + stage2_roi_dims[2]].copy()

            # Estimate Prostate Mask to Mute External Predictions
            mask              = prior
            mask[mask>0.0010] = 1           # Retain Predictions Within 1% Likelihood of the Presence of Prostate
            for i in range(mask.shape[0]): 
                mask[i]   = cv2.GaussianBlur(mask[i], (15,15), cv2.BORDER_DEFAULT)

        # Ensemble Predictions for Benign Class from All Classifiers (Equal Weight)
        for i in range(num_classifiers):
            predictions.append(np.float(prediction_csv[i][scan][2]))
        stage2_class0_probs.append(np.mean(predictions))
        stage2_class1_probs.append(1-np.mean(predictions))
        predictions = []

        # For Each Whole-Volume
        if (patch_counter%stage2_num_patches==0):
            '''
            Decision Fusion: Fuse Dual-Stage Predictions to Single, Penalized Softmax Volume 
            
            If a patch covering the candidate detection is classified as benign by the corresponding classifier(s), we downweight it with
            a penalizing factor ('lambda'), spread across a pre-defined, static weight map ('prior'). This weight map is the weighted
            probabilistic prior of the likelihood of malignancy across prostate zones, generated using all prostate zonal segmentations and
            tumor annotations from the training cases [ref: generate_prior.py]. Finally the octant patches are recomposed into a whole-volume,
            as per the inverse of the original deconstruction algorithm [ref: preprocess_classification_data.py/decompose_3d_octants_plus]. 
            '''
            if (np.min(stage2_class0_probs) > stage2_case_threshold):
                if (stage2_class0_probs[0] > stage2_patch_threshold):
                        print(subject_id)
                        stage2_roi[0:8,0:64,0:64]          = stage2_roi[0:8,0:64,0:64]          * lambda_penalty * prior_roi[0:8,0:64,0:64]
                        stage1_softmax[0:11,0:80,0:80]     = stage1_softmax[0:11,0:80,0:80]     * lambda_penalty * prior[0:11,0:80,0:80]      
                        
                if (stage2_class0_probs[1] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[4:12,0:64,0:64]         = stage2_roi[4:12,0:64,0:64]         * lambda_penalty * prior_roi[4:12,0:64,0:64]        
                        stage1_softmax[7:18,0:80,0:80]     = stage1_softmax[7:18,0:80,0:80]     * lambda_penalty * prior[7:18,0:80,0:80] 

                if (stage2_class0_probs[2] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[0:8,48:112,0:64]        = stage2_roi[0:8,48:112,0:64]        * lambda_penalty * prior_roi[0:8,48:112,0:64]
                        stage1_softmax[0:11,64:144,0:80]   = stage1_softmax[0:11,64:144,0:80]   * lambda_penalty * prior[0:11,64:144,0:80]      

                if (stage2_class0_probs[3] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[4:12,48:112,0:64]       = stage2_roi[4:12,48:112,0:64]       * lambda_penalty * prior_roi[4:12,48:112,0:64]     
                        stage1_softmax[7:18,64:144,0:80]   = stage1_softmax[7:18,64:144,0:80]   * lambda_penalty * prior[7:18,64:144,0:80]

                if (stage2_class0_probs[4] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[0:8,0:64,48:112]        = stage2_roi[0:8,0:64,48:112]        * lambda_penalty * prior_roi[0:8,0:64,48:112]   
                        stage1_softmax[0:11,0:80,64:144]   = stage1_softmax[0:11,0:80,64:144]   * lambda_penalty * prior[0:11,0:80,64:144]

                if (stage2_class0_probs[5] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[4:12,0:64,48:112]       = stage2_roi[4:12,0:64,48:112]       * lambda_penalty * prior_roi[4:12,0:64,48:112]
                        stage1_softmax[7:18,0:80,64:144]   = stage1_softmax[7:18,0:80,64:144]   * lambda_penalty * prior[7:18,0:80,64:144]    

                if (stage2_class0_probs[6] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[0:8,48:112,48:112]      = stage2_roi[0:8,48:112,48:112]      * lambda_penalty * prior_roi[0:8,48:112,48:112]   
                        stage1_softmax[0:11,64:144,64:144] = stage1_softmax[0:11,64:144,64:144] * lambda_penalty * prior[0:11,64:144,64:144]

                if (stage2_class0_probs[7] > stage2_patch_threshold):
                        print(subject_id)  
                        stage2_roi[4:12,48:112,48:112]     = stage2_roi[4:12,48:112,48:112]     * lambda_penalty * prior_roi[4:12,48:112,48:112]
                        stage1_softmax[7:18,64:144,64:144] = stage1_softmax[7:18,64:144,64:144] * lambda_penalty * prior[7:18,64:144,64:144]

                # Integrate Penalized Classifier ROI into Original Stage 1 Softmax Volume
                stage1_softmax[skip_slices_z : skip_slices_z + stage2_roi_dims[0],
                               skip_slices_x : skip_slices_x + stage2_roi_dims[1],
                               skip_slices_y : skip_slices_y + stage2_roi_dims[2]] = stage2_roi

                # Multiply with Prostate Mask to Mute Detections Beyond Prostate
                stage1_softmax = stage1_softmax * (((1-mask)*lambda_penalty) + mask)

            # Export Dual-Stage Predictions and Corresponding Labels
            np.save(dual_stage_save_path+subject_id+'_softmax.npy', stage1_softmax)
            np.save(dual_stage_save_path+subject_id+'_label.npy',   stage1_lbl)
            stage2_class0_probs = []
        patch_counter += 1


def takeSecond(elem):
    return elem[1]


# Compute Prediction Statistics 
def compute_pred_stats(softmax_dir, patch_dims, val_size, threshold=0.50):

    # Compile All Volumes in Single Sweep + Compute List of Thresholds 
    counter         = 0
    all_softmax     = np.zeros(shape=(val_size,patch_dims[0],patch_dims[1],patch_dims[2]), dtype=np.float64)
    all_labels      = np.zeros(shape=(val_size,patch_dims[0],patch_dims[1],patch_dims[2]), dtype=np.uint8)
    all_subject_ids = []
    for f in os.listdir(softmax_dir):
        if '_softmax.npy' in f:
            all_softmax[counter,:,:,:] = np.load(softmax_dir+f)
            all_labels[counter,:,:,:]  = np.load(softmax_dir+f.split('_softmax')[0]+'_label.npy')
            all_subject_ids.append(f.split('_softmax')[0])                   
            counter += 1

    benign_list = []  # Benign Scans
    FN_list     = []  # False Negatives
    fTP_list    = []  # Full True Positives
    pTP_list    = []  # Partial True Positives
    
    # Loop over Images
    for f in range(all_softmax.shape[0]):
        y_list_all            = []   
        y_list_pnp            = []   
        total_normal_patients = 0
        total_patients        = 1        

        # Load and Preprocess Softmax, Label Volumes
        y_true     = all_labels[f,:,:,:].copy()
        y_pred     = all_softmax[f,:,:,:].copy()
        subject_id = all_subject_ids[f]

        y_list_pat, num_lesions_gt = compute_pred_vector(y_pred, y_true)

        # Accumulate Outputs
        y_list_all        +=    y_list_pat
        if num_lesions_gt == 0: y_list_pnp += y_list_pat
        if num_lesions_gt == 0: total_normal_patients += 1

        # Get Lesion-Based Results
        FP_per_image, FP_per_normal_image, sensitivity, thresholds = y_to_FROC(y_list_all, y_list_pnp, total_patients, total_normal_patients,
                                                                               threshold_mode='single', single_threshold=threshold)
        
        if (math.isnan(sensitivity[0])):                benign_list.append(subject_id)  # Benign Scans (Redundancy Check)
        elif (sensitivity[0]==0.0):                     FN_list.append(subject_id)      # False Negatives (Missed the Lesion)
        elif (sensitivity[0]==1.0):                     fTP_list.append(subject_id)     # All True Positives (Caught Everything)
        elif (sensitivity[0]>0.0)&(sensitivity[0]<1.0): pTP_list.append(subject_id)     # Partial True Positives (Caught A Few, Missed A Few)
        
    return benign_list, FN_list, fTP_list, pTP_list
