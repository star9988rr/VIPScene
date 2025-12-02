import numpy as np
import os
import math
import json
import argparse
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from plyfile import PlyData


# Filtering Thresholds
MIN_GROUND_POINTS = 10
MIN_OBJECT_POINTS = 5
MIN_WINDOW_POINTS = 100
BBOX_DOWNSAMPLE_SIZE = 1000

SCENE_BOUNDARY_OFFSET = 0.5


from utils_cr import (
    extract_ground, point_to_plane_distance, rotation_matrix_from_vectors,
    rotate_y, convert_box_vertices, rotate_x_180deg, rotate_point,
    append_invalid, transform_point_cloud, compute_rotation_matrix, get_bbox_2d
)


# ============================================================================
# Data loading functions
# ============================================================================

def process_ground(scene_name, scene_id, bbox_dir_base):
    pseudo_lidar_ground_path = os.path.join(bbox_dir_base, scene_name, scene_id, 'ground_pts3d.npz')
    pseudo_lidar_ground = np.load(pseudo_lidar_ground_path)['arr_0']
    if pseudo_lidar_ground.shape[0] > MIN_GROUND_POINTS:
        return True, extract_ground(pseudo_lidar_ground)
    return False, None


def prepare_vertex(recon_dir_base, scene_name, scene_id, rotation_matrix, ground_equ):
    ply_path = os.path.join(recon_dir_base, scene_name, scene_id, 'obj_scene.ply')
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    points = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(np.float32)
    vertex = points @ rotation_matrix.T
    if ground_equ is not None:
        vertex[:, 1] -= point_to_plane_distance(ground_equ, 0, 0, 0)
    return vertex


def load_object_instances(bbox_dir_base, scene_name, scene_id, rotation_matrix, ground_equ, object_name_list):
    data = np.load(os.path.join(bbox_dir_base, scene_name, scene_id, 'obj_pts3d.npz'))
    obj_instance = [data[key] for key in data.files]
    ground_offset = point_to_plane_distance(ground_equ, 0, 0, 0) if ground_equ is not None else 0
    
    pseudo_lidar_list = []
    for mask_ind, cur_obj in enumerate(obj_instance):
        pseudo_lidar = np.dot(cur_obj, rotation_matrix)
        if pseudo_lidar.shape[0] > 0:
            pseudo_lidar = pseudo_lidar[DBSCAN(eps=0.1, min_samples=20).fit_predict(pseudo_lidar) != -1]
        if ground_equ is not None:
            pseudo_lidar[:, 1] -= ground_offset
        pseudo_lidar_list.append(pseudo_lidar)
    return pseudo_lidar_list


def load_window_instances(bbox_dir_base, scene_name, scene_id, rotation_matrix, ground_equ):
    window_pts3d_path = os.path.join(bbox_dir_base, scene_name, scene_id, 'window_pts3d.npz')
    window_list = []
    if os.path.exists(window_pts3d_path):
        data = np.load(window_pts3d_path)
        obj_instance = [data[key] for key in data.files]
        ground_offset = point_to_plane_distance(ground_equ, 0, 0, 0) if ground_equ is not None else 0
        
        for cur_obj in obj_instance:
            pseudo_lidar = np.dot(cur_obj, rotation_matrix)
            if pseudo_lidar.shape[0] > 0:
                pseudo_lidar = pseudo_lidar[DBSCAN(eps=0.1, min_samples=20).fit_predict(pseudo_lidar) == 0]
            if ground_equ is not None:
                pseudo_lidar[:, 1] -= ground_offset
            window_list.append(pseudo_lidar)
    return window_pts3d_path, window_list


# ============================================================================
# Computation and analysis functions
# ============================================================================

def find_room_orientation(pseudo_lidar_list, min_x, min_z, has_ground, ground_plane):
    orientations, dimensions = [], []
    for pseudo_lidar in pseudo_lidar_list:
        if pseudo_lidar.shape[0] < MIN_OBJECT_POINTS:
            continue
        pseudo_lidar_adj = pseudo_lidar.copy()
        pseudo_lidar_adj[:, [0, 2]] -= [min_x, min_z]
        pseudo_lidar_adj = rotate_point(pseudo_lidar_adj)
        bbox_params = estimate_bbox(pseudo_lidar_adj, None, ground_plane if has_ground else None)
        orientations.extend(bbox_params[4])
        dimensions.extend(bbox_params[2])

    valid_degrees = [deg % 180 for deg, dim in zip(orientations, dimensions) if dim != [-1, -1, -1]]
    if not valid_degrees:
        print('No valid objects found, defaulting to 0 degrees orientation')
        return 0
    
    orientation_counts = {}
    for deg in valid_degrees:
        base_deg = next((b for b in orientation_counts if abs(deg - b) <= 10), deg)
        orientation_counts[base_deg] = orientation_counts.get(base_deg, 0) + 1
    
    room_orientation = max(orientation_counts.items(), key=lambda x: x[1])[0]
    print(f'Room orientation: {room_orientation} degrees')
    return room_orientation % 90


def estimate_bbox(rotated_pc, prior, ground_plane=None):
    # Subsample input point cloud if needed
    if rotated_pc.shape[0] > BBOX_DOWNSAMPLE_SIZE:
        rotated_pc = rotated_pc[np.random.randint(0, rotated_pc.shape[0], BBOX_DOWNSAMPLE_SIZE)]

    pca = PCA(2)
    pca.fit(rotated_pc[:, [0, 2]])
    yaw_vec = pca.components_[0, :]
    yaw = np.arctan2(yaw_vec[1], yaw_vec[0])
    yaw_degree = np.degrees(yaw)

    # Rotate the point cloud to align with the x-axis and z-axis
    pc_aligned = rotate_y(yaw) @ rotated_pc.T
    mins, maxs = pc_aligned.min(axis=1), pc_aligned.max(axis=1)
    dx, dy, dz = maxs - mins
    cx, cy, cz = (mins + maxs) / 2

    # Convert vertices and rotate back
    vertices = convert_box_vertices(cx, cy, cz, dx, dy, dz, 0).astype(np.float16)
    vertices = np.dot(rotate_y(-yaw), vertices.T).T
    
    return ([vertices], [vertices.mean(0)], [[dz, dy, dx]], 
            [np.array([rotate_y(-yaw + i * math.pi/2) for i in range(4)])], [yaw_degree])


def process_point_cloud_list(pc_list_corrected, min_x_corrected, max_z_corrected, has_ground, ground_plane):
    """Process a list of point clouds to generate bounding boxes."""
    boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list, new_pc_list = [], [], [], [], [], []
    
    for pseudo_lidar in pc_list_corrected:
        if pseudo_lidar.shape[0] < MIN_OBJECT_POINTS:
            append_invalid([boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list, new_pc_list])
            continue
        
        pseudo_lidar[:, [0, 2]] -= [min_x_corrected, max_z_corrected + SCENE_BOUNDARY_OFFSET]
        new_pc_list.append(pseudo_lidar)
        bbox_params = estimate_bbox(pseudo_lidar, None, ground_plane if has_ground else None)
        for i, result_list in enumerate([boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list]):
            result_list.extend(bbox_params[i])
    
    return new_pc_list, boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list


# ============================================================================
# Main processing function
# ============================================================================

def process_instances(scene_name, scene_id, has_ground, ground_equ, recon_dir_base, bbox_dir_base, json_dir_base):
    """Process each instance to generate 3D bounding boxes."""
    # Compute rotation matrix
    rotation_matrix, ground_equ = compute_rotation_matrix(ground_equ)
    ground_plane = np.array([0, -1, 0, point_to_plane_distance(ground_equ, 0, 0, 0)]) if ground_equ is not None else None

    # Load data
    vertex = prepare_vertex(recon_dir_base, scene_name, scene_id, rotation_matrix, ground_equ)
    with open(os.path.join(json_dir_base, scene_name, scene_id, 'map_name.json'), 'r') as f:
        object_name_list = json.load(f)
    pseudo_lidar_list = load_object_instances(bbox_dir_base, scene_name, scene_id, rotation_matrix, ground_equ, object_name_list)
    window_pts3d_path, window_list = load_window_instances(bbox_dir_base, scene_name, scene_id, rotation_matrix, ground_equ)
    
    # Find room orientation
    pseudo_lidar_whole = np.concatenate(pseudo_lidar_list)
    min_x, min_z = pseudo_lidar_whole[:, 0].min(), pseudo_lidar_whole[:, 2].min()
    room_orientation = find_room_orientation(pseudo_lidar_list, min_x, min_z, has_ground, ground_plane)
    
    # Transform point clouds
    vertex[:, [0, 2]] -= [min_x, min_z]
    vertex = (rotate_y(np.radians(room_orientation)) @ rotate_point(vertex).T).T
    obj_pc_list = [transform_point_cloud(pc.copy(), min_x, min_z, room_orientation) for pc in pseudo_lidar_list]
    win_pc_list = [transform_point_cloud(pc.copy(), min_x, min_z, room_orientation) for pc in window_list]
    
    # Calculate room size
    pseudo_lidar_whole = np.concatenate(obj_pc_list)
    min_x_corrected, max_z_corrected = pseudo_lidar_whole[:, 0].min(), pseudo_lidar_whole[:, 2].max()
    length = round(pseudo_lidar_whole[:, 0].ptp(), 2)
    width = round(pseudo_lidar_whole[:, 2].ptp(), 2) + SCENE_BOUNDARY_OFFSET
    print(f'room size: {length} x {width}')

    # Process objects
    new_pseudo_lidar_list, boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list = \
        process_point_cloud_list(obj_pc_list, min_x_corrected, max_z_corrected, has_ground, ground_plane)
    
    # Process windows
    new_window_list, window_boxes3d, window_center_cam_list, window_dimension_list, window_R_cam_list, window_degree_list = \
        ([], [], [], [], [], []) if not os.path.exists(window_pts3d_path) else \
        process_point_cloud_list(win_pc_list, min_x_corrected, max_z_corrected, has_ground, ground_plane)

    vertex[:, [0, 2]] -= [min_x_corrected, max_z_corrected + SCENE_BOUNDARY_OFFSET]
    return (new_pseudo_lidar_list, boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list, length, width,
            new_window_list, window_boxes3d, window_center_cam_list, window_dimension_list, window_R_cam_list, window_degree_list)


# ============================================================================
# JSON generation functions
# ============================================================================

def build_valid_objects(map_name, center_cam_list, dimension_list, degree_list):
    valid_objects = {}
    for obj_id, obj_name in map_name.items():
        idx = int(obj_id)
        if idx >= len(dimension_list):
            raise ValueError(f"Object index {idx} exceeds dimension list length")
        if dimension_list[idx] != [-1, -1, -1] and center_cam_list[idx].tolist() != [-1, -1, -1]:
            valid_objects[idx] = {
                "description": f"a {obj_name}",
                "location": "floor",
                "size": dimension_list[idx],
                "position": center_cam_list[idx].tolist(),
                "degree": degree_list[idx],
                "quantity": 1,
                "variance_type": "same",
                "objects_on_top": [],
                "object_name": obj_name
            }
    return valid_objects


def sort_objects_by_area(valid_objects):
    return dict(sorted(valid_objects.items(), key=lambda x: x[1]["size"][0] * x[1]["size"][2], reverse=True))


def is_stacked(obj1_bbox, obj2_bbox, pos1, pos2, size1, size2, y1_top, y2_bottom):
    """Check if obj2 is stacked on obj1."""
    # Check if vertical relationship is valid (obj2 should be above obj1)
    if y2_bottom <= y1_top - 0.1:
        return False

    # Check if obj1 (bottom object) is a large object
    area1 = (obj1_bbox['x_max'] - obj1_bbox['x_min']) * (obj1_bbox['z_max'] - obj1_bbox['z_min'])
    if area1 < 0.2:
        return False
    
    # Check overlap in x-z plane
    overlap_x = max(0, min(obj1_bbox['x_max'], obj2_bbox['x_max']) - max(obj1_bbox['x_min'], obj2_bbox['x_min']))
    overlap_z = max(0, min(obj1_bbox['z_max'], obj2_bbox['z_max']) - max(obj1_bbox['z_min'], obj2_bbox['z_min']))
    
    if overlap_x > 0 and overlap_z > 0:
        # Overlapping case
        area1 = (obj1_bbox['x_max'] - obj1_bbox['x_min']) * (obj1_bbox['z_max'] - obj1_bbox['z_min'])
        area2 = (obj2_bbox['x_max'] - obj2_bbox['x_min']) * (obj2_bbox['z_max'] - obj2_bbox['z_min'])
        overlap_area = overlap_x * overlap_z
        smaller_area = min(area1, area2)
        return overlap_area > 0.1 * smaller_area
    
    return False


def detect_stacked_objects(valid_objects):
    excluded_names = {'countertop', 'table', 'island', 'sink'}
    
    for idx1 in valid_objects:
        obj1 = valid_objects[idx1]
        pos1 = obj1["position"]
        size1 = obj1["size"]
        bbox1 = get_bbox_2d(pos1, size1)
        y1_top = pos1[1] + size1[1] / 2
        
        for idx2 in valid_objects:
            if idx1 == idx2:
                continue
            
            obj2 = valid_objects[idx2]
            pos2 = obj2["position"]
            size2 = obj2["size"]
            bbox2 = get_bbox_2d(pos2, size2)
            y2_bottom = pos2[1] - size2[1] / 2

            if is_stacked(bbox1, bbox2, pos1, pos2, size1, size2, y1_top, y2_bottom):
                obj_name2 = valid_objects[idx2]['object_name']
                if obj_name2 not in excluded_names:
                    obj1["objects_on_top"].append(f"{idx2}_{obj_name2}")
                    valid_objects[idx2]["location"] = "top"
                    print(f"{obj_name2} is on top of {obj1['object_name']}")


def to_scene_object_dict(valid_objects):
    return {f"{idx}_{obj['object_name']}": obj for idx, obj in valid_objects.items()}


def generate_scenejson(map_name, center_cam_list, dimension_list, degree_list):
    scene_dict = {}
    scene_dict["object"] = {}

    valid_objects = build_valid_objects(map_name, center_cam_list, dimension_list, degree_list)
    valid_objects = sort_objects_by_area(valid_objects)
    for obj in valid_objects.values():
        print(obj["object_name"])
    detect_stacked_objects(valid_objects)
    scene_dict["object"] = to_scene_object_dict(valid_objects)
    return scene_dict


def generate_scenejson_window(map_name, center_cam_list, dimension_list, degree_list, window_list):
    window_dict = {}

    for idx in range(len(center_cam_list)):
        if dimension_list[idx] != [-1, -1, -1] and center_cam_list[idx].tolist() != [-1, -1, -1]:
            if window_list[idx].shape[0] < MIN_WINDOW_POINTS:
                continue
            window_dict[f"{idx}_window"] = {
                "description": f"a window",
                "location": "floor",
                "size": dimension_list[idx],
                "position": center_cam_list[idx].tolist(),
                "degree": degree_list[idx],
                "quantity": 1,
                "variance_type": "same",
                "objects_on_top": [],
                "object_name": 'window'
            }
    
    return window_dict


# ============================================================================
# Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scene_dir", required=True, help='scene directory')
    args = parser.parse_args()

    scene_dir = args.scene_dir
    scene_name = os.path.basename(scene_dir)
    base_dir = os.path.dirname(os.path.dirname(scene_dir))
    
    bbox_dir = os.path.join(base_dir, 'bbox_3d')
    recon_dir = os.path.join(base_dir, 'obj_reconstruction')
    json_dir = os.path.join(base_dir, 'json_files')

    scene_paths = sorted([os.path.join(scene_dir, x) for x in os.listdir(scene_dir) 
                          if os.path.isdir(os.path.join(scene_dir, x))])

    for scene_path in scene_paths:
        scene_id = os.path.basename(scene_path)

        with open(os.path.join(json_dir, scene_name, scene_id, 'map_name.json'), 'r') as file:
            map_name = json.load(file)

        # Process floor data and estimate ground plane
        has_ground, ground_equ = process_ground(scene_name, scene_id, bbox_dir)

        # Process instances and generate 3D bounding boxes
        (pseudo_lidar_list, boxes3d, center_cam_list, dimension_list, R_cam_list, degree_list, length, width,
         window_list, window_boxes3d, window_center_cam_list, window_dimension_list, window_R_cam_list, window_degree_list) = process_instances(
            scene_name, scene_id, has_ground, ground_equ, recon_dir, bbox_dir, json_dir)

        # Save 3D bounding boxes
        base_path = os.path.join(bbox_dir, scene_name, scene_id)
        save_data = {
            'pseudo_lidar.npy': np.concatenate(pseudo_lidar_list),
            'bbox_3d.npy': np.array(boxes3d),
            'window_bbox_3d.npy': np.array(window_boxes3d),
            'R_cam.npy': np.stack(R_cam_list)
        }
        for filename, data in save_data.items():
            np.save(f'{base_path}/{filename}', data)
        np.savez(f'{base_path}/obj_pts3d_rescaled.npz', *pseudo_lidar_list)
        np.savez(f'{base_path}/window_pts3d_rescaled.npz', *window_list)

        assert len(map_name) == len(dimension_list)
        for i in range(len(dimension_list)):
            print(map_name[str(i)], dimension_list[i])

        scene_data = generate_scenejson(map_name, center_cam_list, dimension_list, degree_list)
        scene_data['room_size'] = [length, width]
        window_data = generate_scenejson_window(map_name, window_center_cam_list, window_dimension_list, window_degree_list, window_list)
        scene_data['window'] = window_data

        # Save results
        with open(os.path.join(json_dir, scene_name, scene_id, 'scene.json'), "w") as f:
            json.dump(scene_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
