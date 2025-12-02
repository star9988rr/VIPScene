from typing import Any

import numpy as np
from collections import defaultdict
import open3d as o3d


def fps(points, num_samples):
    N = points.shape[0]
    sampled_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(1, num_samples):
        dist = np.linalg.norm(points - points[sampled_indices[-1]], axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices.append(np.argmax(distances))

    return points[sampled_indices]


def rotate_180deg(q, mode='x', is_point=True):
    if mode=='x':
        R = np.array(
            [[1, 0, 0],
             [0, -1, 0],
             [0, 0, -1]]
        )
    elif mode=='y':
        R = np.array(
            [[-1, 0, 0],
             [0, 1, 0],
             [0, 0, -1]]
        )
    elif mode=='z':
        R = np.array(
            [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]]
        )
    else:
        raise NotImplementedError
    if is_point:
        rotated_q = np.matmul(q, R)
    else:
        rotated_q = np.matmul(R, q)
    return rotated_q

def extract_yaw_from_rotation_matrix(R):
    theta = np.arctan2(R[2, 0], R[2, 2])
    return theta



def match_clouds(points_dict, candidate_ids_dict, segments_dict, init_R_dict):
    Rt_dict = defaultdict(list)
    for name in segments_dict.keys():
        pc_seg = segments_dict[name]
        
        for candidate_id, pc_candidate in zip(candidate_ids_dict[name], points_dict[name]):
            
            target_points = pc_candidate
            Rt, rmse_list = calculate_rt(target_points, pc_seg, init_R_dict[name], cal_mode='all_rmse')
            results = [adjust_matrix(rt['R'], rt['t']) for rt in Rt]
            Rt_list, dist_list = zip(*[(r[1], r[0]) for r in results])
            
            Rt_dict[name].append((list(Rt_list), rmse_list, list[Any](dist_list), candidate_id))
        
        if not Rt_dict[name]:
            print(f"no {name} points were recorded in the database.")
    
    return Rt_dict


def calculate_icp(source_pc, target_pc, init_trans_guess=None, threshold=None):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pc)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pc)
    
    trans_init = init_trans_guess if init_trans_guess is not None else np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    return reg_p2p.transformation, reg_p2p.inlier_rmse


def calculate_rt(source, target, init_R_list, threshold=50, cal_mode='lowest_rmse'):
    t_init = np.mean(target, axis=0) - np.mean(source, axis=0)
    rmse_list = []
    Rt_list = []
    
    for init_R in init_R_list:
        init_matrix = np.eye(4)
        init_matrix[:3, :3] = init_R
        init_matrix[:3, 3] = t_init
        
        icp_matrix, rmse = calculate_icp(source, target, init_trans_guess=init_matrix, threshold=threshold)
        rmse_list.append(rmse)
        Rt_list.append({'R': icp_matrix[:3, :3], 't': icp_matrix[:3, 3]})
    
    if cal_mode == 'lowest_rmse':
        idx = np.argmin(rmse_list)
        return Rt_list[idx], rmse_list[idx]
    return Rt_list, np.array(rmse_list)

def adjust_matrix(R, t, mode='y'):
    mode_map = {'x': (0, np.array([1, 0, 0])), 'y': (1, np.array([0, 1, 0])), 'z': (2, np.array([0, 0, 1]))}
    if mode not in mode_map:
        raise ValueError("mode must be one of 'x', 'y', or 'z'.")
    
    ignore_index, ref_axis = mode_map[mode]
    R_axis = R[:, ignore_index] / np.linalg.norm(R[:, ignore_index])
    rotation_axis = np.cross(R_axis, ref_axis)
    
    if np.linalg.norm(rotation_axis) < 1e-8:
        adjusted_R = R
    else:
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(R_axis, ref_axis), -1.0, 1.0))
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        adjusted_R = (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K) @ R
    
    t_new = t.copy()
    t_new[ignore_index] = 0
    return 1 - np.dot(R_axis, adjusted_R[:, ignore_index]), {'R': adjusted_R, 't': t_new}
