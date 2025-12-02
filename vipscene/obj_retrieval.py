import os
import numpy as np
import json
import trimesh
import open_clip
from shapely.geometry import box
from shapely.affinity import rotate
from vipscene.generation.objaverse_retriever import ObjathorRetriever
from sentence_transformers import SentenceTransformer
from vipscene.generation.utils import get_annotations, get_bbox_dims
from vipscene.generation.utils_matching import fps, rotate_180deg, extract_yaw_from_rotation_matrix, match_clouds
from vipscene.constants import OBJATHOR_OBJ_DIR
from tqdm import tqdm
import argparse
import time


def calculate_iou(box1, box2):
    """Calculate the intersection over union (IoU) of two boxes."""
    x1, y1, w1, h1, angle1 = box1
    x2, y2, w2, h2, angle2 = box2

    # Create the boxes as polygons
    rect1 = box(x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2)
    rect2 = box(x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2)

    # Rotate the boxes
    rect1 = rotate(rect1, angle1, use_radians=False)
    rect2 = rotate(rect2, angle2, use_radians=False)

    # Calculate the intersection and union
    intersection = rect1.intersection(rect2).area
    union = rect1.union(rect2).area

    return intersection / union


def load_scene_data(scene_name, base_path, bbox_base_path):
    scene_dir = os.path.join(bbox_base_path, scene_name)
    pc_data = np.load(os.path.join(scene_dir, 'obj_pts3d_rescaled.npz'))
    pc_data = [pc_data[key] for key in pc_data.files]
    
    bbox_data = np.load(os.path.join(scene_dir, 'bbox_3d.npy'))
    init_R_data = np.load(os.path.join(scene_dir, 'R_cam.npy'))
    
    with open(os.path.join(base_path, scene_name, 'scene.json'), 'r') as f:
        scene_json = json.load(f)
    
    return pc_data, bbox_data, init_R_data, scene_json, scene_json['object']


def retrieve_candidates(obj_retriever, obj_visual_features, num, name, retrieval_threshold):
    """Retrieve candidate objects using visual features or text."""
    if obj_visual_features is not None:
        visual_feature = obj_visual_features[num]
        return obj_retriever.retrieve_unified(queries=[f"a 3D model of {name}"], visual_feature=visual_feature, threshold=retrieval_threshold)
    else:
        return obj_retriever.retrieve([f"a 3D model of {name}"], retrieval_threshold)


def filter_candidates(candidates, database):
    """Filter candidates based on annotations."""
    filtered = [
        candidate
        for candidate, annotation in zip(
            candidates,
            [get_annotations(database[candidate[0]]) for candidate in candidates],
        )
        if annotation["onFloor"]
        and (not annotation["onCeiling"])
        and all(k not in annotation["category"].lower() for k in ["door", "window", "frame"])
    ]
    return filtered[:3]


def select_best_candidate_by_size(candidates, database, target_size):
    if not candidates:
        return []
    
    target_xz = np.array([max(target_size[0], target_size[2]), min(target_size[0], target_size[2])])
    sizes = [get_bbox_dims(database[c[0]]) for c in candidates]
    candidate_xz = np.array([[max(s['x'], s['z']), min(s['x'], s['z'])] for s in sizes])
    
    return [candidates[np.argmin(np.abs(candidate_xz - target_xz).mean(axis=1))]]


def load_candidate_meshes(candidates, num_fps_points, num_points):
    asset_ids = []
    obj_points = []
    
    for asset_id, _ in candidates:
        mesh = trimesh.load(os.path.join(OBJATHOR_OBJ_DIR, f'{asset_id}.obj'))
        points = fps(np.array(mesh.sample(num_points)), num_fps_points)
        asset_ids.append(asset_id)
        obj_points.append(points)
    
    return asset_ids, obj_points


def find_best_matches(rel_dict, plan, database):
    """Find best matches based on IoU and RMSE."""
    best_name_Rt_dict = {}
    
    for name, val in rel_dict.items():
        target = plan[name]
        target_size, target_degree, target_position = target['size'], target['degree'], target['position']
        
        best_match = None
        best_loss = float('inf')
        
        for rt_list, rmse_list, _, asset_id in val:
            candidate_size = get_bbox_dims(database[asset_id])
            
            for rt, rmse in zip(rt_list, rmse_list):
                R_new = rotate_180deg(rt['R'], is_point=False)
                candidate_degree = np.degrees(extract_yaw_from_rotation_matrix(R_new))
                
                target_box = (target_position[0], target_position[2], target_size[0], target_size[2], target_degree)
                candidate_box = (target_position[0], target_position[2], candidate_size['x'], candidate_size['z'], candidate_degree)
                
                loss = calculate_iou(target_box, candidate_box) + rmse
                
                if loss < best_loss:
                    best_loss = loss
                    best_match = (rt, rmse, None, asset_id)
        
        best_name_Rt_dict[name] = best_match
    
    return best_name_Rt_dict


def update_and_save_plan(plan, best_name_Rt_dict, name_center_dict, database, base_path, scene_name, scene_json):
    """Update plan with best matches and save to file."""
    for obj in plan:
        rt, _, _, asset_id = best_name_Rt_dict[obj]
        R_new = rotate_180deg(rt['R'], is_point=False)
        t_new = rotate_180deg(name_center_dict[obj], is_point=False)
        
        plan[obj].update({
            'degree': np.degrees(extract_yaw_from_rotation_matrix(R_new)),
            'position': t_new.tolist(),
            'asset_id': asset_id,
            'dimension': get_bbox_dims(database[asset_id])
        })
    
    output_path = os.path.join(base_path, f'{scene_name}/scene_with_assets.json')
    with open(output_path, 'w') as json_file:
        json.dump(scene_json, json_file, indent=4)


def process_scene(path, args, obj_retriever, database, retrieval_threshold):
    """Process a single scene."""
    start_time = time.time()
    scene_name = path.split('/')[-1].split('.')[0]
    base_path = args.input_dir.replace('image_data', 'json_files')
    bbox_base_path = args.input_dir.replace('image_data', 'bbox_3d')
    
    # Load visual features for query objects
    visual_features_path = os.path.join(bbox_base_path, scene_name, 'clip_feature.npy')
    obj_visual_features = np.load(visual_features_path)
    
    num_fps_points = 1000
    num_points = 10000
    
    # Load scene data
    pc_data, bbox_data, init_R_data, scene_json, plan = load_scene_data(scene_name, base_path, bbox_base_path)
    
    name_seg_dict = {}
    name_init_R_dict = {}
    name_ids_dict = {}
    name_points_dict = {}
    name_center_dict = {}
    
    for object_name in plan:
        num = int(object_name.split('_')[0])
        obj_pt3d = pc_data[num]
        if not obj_pt3d.shape[0]:
            continue
        
        name = object_name.split('_')[-1]
        name_seg_dict[object_name] = fps(obj_pt3d, num_fps_points)
        name_center_dict[object_name] = np.mean(bbox_data[num], axis=0)
        name_init_R_dict[object_name] = init_R_data[num]
        
        candidates = filter_candidates(
            retrieve_candidates(obj_retriever, obj_visual_features, num, name, retrieval_threshold),
            database
        )
        if not candidates:
            print(f"No candidates found for {name}")
            continue
        
        candidates = select_best_candidate_by_size(candidates, database, plan[object_name]['size'])
        name_ids_dict[object_name], name_points_dict[object_name] = load_candidate_meshes(
            candidates, num_fps_points, num_points
        )
    
    print(f"First processing for scene {scene_name} took: {time.time() - start_time:.2f} seconds")
    
    rel_dict = match_clouds(name_points_dict, name_ids_dict, name_seg_dict, name_init_R_dict)
    best_name_Rt_dict = find_best_matches(rel_dict, plan, database)
    update_and_save_plan(plan, best_name_Rt_dict, name_center_dict, database, base_path, scene_name, scene_json)
    
    print(f"Total processing for scene {scene_name} took: {time.time() - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Object retrieval for scene generation")
    parser.add_argument("--input_dir", required=True, help='Input directory path')
    args = parser.parse_args()

    # Initialize CLIP
    (clip_model, _, clip_preprocess) = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # Initialize SBERT
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    retrieval_threshold = 31

    obj_retriever = ObjathorRetriever(clip_model, clip_preprocess, clip_tokenizer, sbert_model, retrieval_threshold)
    database = obj_retriever.database

    scene_files = [os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir) 
                   if os.path.isdir(os.path.join(args.input_dir, x)) and not x.endswith('.DS_Store')]
    scene_files.sort()

    for path in tqdm(scene_files):
        process_scene(path, args, obj_retriever, database, retrieval_threshold)


if __name__ == "__main__":
    main()
