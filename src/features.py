"""Several functions for feature engineering"""
import numpy as np
import matplotlib.pyplot as plt
import math


def get_location_percentage(x):
    """For e.g. the x locations, get the percentage along the curve"""
    return np.arange(len(x)) / len(x)


def get_clock(x, y):
    center_x = np.mean(x)
    center_y = np.mean(y)
    clock = []
    for idx in range(len(x)):
        angle = math.atan2(x[idx] - center_x, y[idx] - center_y)
        clock.append(angle)
    return clock


def get_motion_map(cavity_dfs, rest_dfs):
    for i in range(len(cavity_dfs)):
        total_df = cavity_dfs[i] + rest_dfs[i]
        total_df = np.linalg.norm(total_df, axis=2)
        if i == 0:
            mean_df = total_df
        else:
            mean_df += total_df
    return mean_df / len(cavity_dfs)


def get_local_motion(cavity_dfs, rest_dfs, contour):
    """Evaluate average local motion for each pixel along contour"""
    motion_map = get_motion_map(cavity_dfs, rest_dfs)
    local_motion = []
    for i in range(len(contour.x)):
        local_motion.append(motion_map[int(contour.y[i]), int(contour.x[i])])
    return local_motion


def get_motion_stats(cavity_dfs, rest_dfs):
    """Evaluate average, min, max motion for entire movie"""
    motion_map = get_motion_map(cavity_dfs, rest_dfs)
    mean = np.mean(motion_map)
    max = np.max(motion_map)
    return mean, max


def get_registration_based_features(vs_computation_input, contour):
    """Get features based on registration results"""
    average_motion, max_motion = get_motion_stats(
        vs_computation_input["cavity_dfs"], vs_computation_input["rest_dfs"]
    )
    local_motion = get_local_motion(
        vs_computation_input["cavity_dfs"], vs_computation_input["rest_dfs"], contour
    )
    return average_motion, max_motion, local_motion


def get_contour_based_features(contour):
    """Get features based on contour"""
    # location_percentage = get_location_percentage(contour.x)
    curvature = contour.curvature
    clock = get_clock(contour.x, contour.y)
    return curvature, clock
