import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
import os
import numpy as np
import trimesh
import cv2
import argparse
import tempfile
from pathlib import Path
import json
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--scene_dir", type=str, required=True,
                      help="Base scene directory containing mask, depth, json_files, image_data subdirectories")
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="Output directory")
    parser.add_argument("--min_conf_thr", type=float, default=10.0,
                      help="Minimum confidence threshold (0-100)")
    parser.add_argument("--mode", type=str, choices=['ground', 'window', 'default'], default='default',
                      help="Mode: 'ground' for ground mode, 'window' for window mode, 'default' for default mode")

    return parser

def find_min_max_indices(mask):
    indices = np.argwhere(mask == 1)
    return (*np.min(indices, axis=0), *np.max(indices, axis=0))

def adaptive_erode_mask(mask, th, k_vertical_min, k_vertical_max, k_horizontal_min, k_horizontal_max):
    """Erode masks with kernel iterations scaled by region size."""
    new_mask = np.zeros_like(mask)
    kernel_vertical = np.ones((3, 1), np.uint8)
    kernel_horizontal = np.ones((1, 3), np.uint8)
    
    for i in range(mask.shape[0]):
        mask_i = mask[i]
        if np.sum(mask_i) == 0:
            new_mask[i] = mask_i
            continue
        
        min_row, min_col, max_row, max_col = find_min_max_indices(mask_i)
        height, width = max_row - min_row, max_col - min_col
        
        # Calculate dynamic kernel iterations based on region dimensions
        k_v = min(k_vertical_min + max(0, (height - th) // 15), k_vertical_max)
        k_h = min(k_horizontal_min + max(0, (width - th) // 15), k_horizontal_max)
        
        eroded_mask_vertical = cv2.erode(mask_i, kernel_vertical, iterations=k_v)
        eroded_mask_horizontal = cv2.erode(mask_i, kernel_horizontal, iterations=k_h)
        new_mask[i] = np.logical_and(eroded_mask_vertical, eroded_mask_horizontal).astype(np.uint8)
    
    return new_mask

def _resize_mask(mask, long_edge_size):
    B, C, H, W = mask.shape  # Get batch_size and mask dimensions
    S = max(H, W)
    
    new_size = long_edge_size
        
    new_H, new_W = tuple(int(round(x * new_size / S)) for x in (H, W))

    resized_masks = np.zeros((B, C, new_H, new_W), dtype=mask.dtype)
    for i in range(B):
        # Use INTER_LANCZOS4 for upsampling and INTER_AREA for downsampling
        interpolation = cv2.INTER_LANCZOS4 if S < new_size else cv2.INTER_AREA
        resized_masks[i, 0] = cv2.resize(mask[i, 0], (new_W, new_H), interpolation=interpolation)
    return resized_masks

def center_crop_mask(mask, cx, cy, square_ok):
    B, C, H, W = mask.shape
    
    # Adjust to be multiple of 16 (8 pixels on each side)
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8
    
    # Handle non-square images
    if not square_ok and W == H:
        halfh = int(3 * halfw / 4)
    
    cropped_masks = np.zeros((B, C, 2 * halfh, 2 * halfw), dtype=mask.dtype)
    for i in range(B):
        cropped_masks[i, 0] = mask[i, 0, cy-halfh:cy+halfh, cx-halfw:cx+halfw]
    return cropped_masks

def plot_3d_points_with_colors(preds, views, depth_maps, min_conf_thr_percentile=80, export_ply_path=None):
    all_points = []
    all_colors = []

    depth_maps = np.concatenate(depth_maps, axis=0).squeeze(1)
    scales = []
    for i, pred in enumerate(preds):
        pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()  
        conf = pred['conf'].cpu().numpy().squeeze()

        conf_thr = np.percentile(conf, min_conf_thr_percentile)

        mask = conf > conf_thr

        depth_pred = pts3d[..., 2]
        depth_true = depth_maps[i]
        valid = mask & (depth_true > 0)
        if valid.sum() == 0:
            continue
        ratios = depth_true[valid] / depth_pred[valid]
        scales.append(np.median(ratios))
    
    s = np.median(scales) if len(scales) > 0 else 1.0

    for i, pred in enumerate(preds):
        pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()
        pts3d = pts3d * s
        
        img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)  
        img_rgb = (img_rgb + 1) * 0.5
        conf = pred['conf'].cpu().numpy().squeeze()

        conf_thr = np.percentile(conf, min_conf_thr_percentile)
        x, y, z = pts3d[..., 0].flatten(), pts3d[..., 1].flatten(), pts3d[..., 2].flatten()
        r, g, b = img_rgb[..., 0].flatten(), img_rgb[..., 1].flatten(), img_rgb[..., 2].flatten()
        conf_flat = conf.flatten()

        mask = conf_flat > conf_thr
        x, y, z = x[mask], y[mask], z[mask]
        r, g, b = r[mask], g[mask], b[mask]

        all_points.append(np.vstack([x, y, z]).T)
        all_colors.append(np.vstack([r, g, b]).T)


    if export_ply_path:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
        point_cloud.export(export_ply_path)
        
    return all_colors, s

def get_3D_obj_from_scene(preds, views, mask, scale, min_conf_thr_percentile=80):
    """Extract object point cloud from a reconstructed scene given masks."""
    all_points = []

    mask_object = np.array(mask).astype(bool)  # Shape (N, 1, 720, 1280)
    # Loop through each set of points in preds
    for i, pred in enumerate(preds):
        if i >= len(mask_object) or mask_object[i].size == 0:
            continue

        pts3d = pred['pts3d_in_other_view'].cpu().numpy().squeeze()
        pts3d = pts3d * scale
        
        conf = pred['conf'].cpu().numpy().squeeze()

        conf_thr = np.percentile(conf, min_conf_thr_percentile)
        x, y, z = pts3d[..., 0].flatten(), pts3d[..., 1].flatten(), pts3d[..., 2].flatten()
        conf_flat = conf.flatten()

        if conf_flat.shape == mask_object[i].flatten().shape:
            mask = conf_flat > conf_thr
            mask = mask & mask_object[i].flatten()
            x, y, z = x[mask], y[mask], z[mask]

            all_points.append(np.vstack([x, y, z]).T)
        else:
            print(f"Warning: Shape mismatch in get_3D_obj_from_scene view {i}. Skipping.")

    return all_points

def aggregate_clip_features_for_object(clip_features_list, mask_list_obj, labels_by_frame, obj):
    """Aggregate CLIP features for an object across views (mean over valid)."""
    aggregated_features = []
    
    for i, clip_features in enumerate(clip_features_list):
        if obj in labels_by_frame[i]:
            # Get the index of this object in the current view
            obj_index = labels_by_frame[i].index(obj)

            # Get corresponding CLIP features based on object index
            if obj_index < len(clip_features):
                aggregated_features.append(clip_features[obj_index])
            else:
                print(f"Warning: Object {obj} index {obj_index} out of range for CLIP features in view {i}")
    
    if len(aggregated_features) == 0:
        return np.zeros_like(clip_features_list[0][0]) if clip_features_list and len(clip_features_list[0]) > 0 else np.zeros(768)
    aggregated_features = np.array(aggregated_features)
    return np.mean(aggregated_features, axis=0)

def build_mask_list_for_object(obj, input_files, mask_list, labels_by_frame):
    """Return per-view binary masks for a given object; zeros if absent."""
    masks = []
    for i in range(len(input_files)):
        mask = mask_list[i] # mask is (N, H, W) or (0, H, W)
        
        # Check if labels_by_frame[i] exists (it might be [])
        if i < len(labels_by_frame) and obj in labels_by_frame[i]:
            obj_index = labels_by_frame[i].index(obj)
            # Check if mask has enough elements
            if obj_index < mask.shape[0]:
                masks.append(mask[obj_index])
            else:
                # This case (label exists but mask doesn't) might happen if padding is misaligned
                # We must append zeros of the correct shape.
                print(f"Warning: Mask index {obj_index} out of bounds for view {i}. Appending empty mask.")
                masks.append(np.zeros_like(mask[0]) if mask.shape[0] > 0 else np.zeros(mask.shape[1:], dtype=mask.dtype))
        else:
            # Object not in this frame, append zeros_like mask[0]
            # Handle (0, H, W) shape
            masks.append(np.zeros_like(mask[0]) if mask.shape[0] > 0 else np.zeros(mask.shape[1:], dtype=mask.dtype))
    return masks

def extract_object_colors_from_views(preds, views, mask_list_obj, min_conf_thr):
    """Extract RGB colors for an object across all views."""
    colors = []
    for i, pred in enumerate(preds):
        # Ensure mask_list_obj[i] exists and is valid
        if i >= len(mask_list_obj) or mask_list_obj[i].size == 0:
            continue

        img_rgb = views[i]['img'].cpu().numpy().squeeze().transpose(1, 2, 0)
        img_rgb = (img_rgb + 1) * 0.5
        conf = pred['conf'].cpu().numpy().squeeze()
        conf_thr = np.percentile(conf, min_conf_thr)
        
        # Ensure shapes match
        if conf.flatten().shape == mask_list_obj[i].flatten().shape:
            mask_flat = (conf.flatten() > conf_thr) & mask_list_obj[i].flatten().astype(bool)
            rgb = img_rgb.reshape(-1, 3)[mask_flat]
            colors.append(rgb)
        else:
            print(f"Warning: Shape mismatch in extract_object_colors_from_views view {i}. Skipping.")

    return np.concatenate(colors, axis=0) if colors else np.zeros((0, 3))

def load_masks(filelist, mask_dir, image_size, depth_maps):
    """Load and process masks from files."""
    mask_list = []
    input_files = sorted(os.listdir(filelist))
    for i in tqdm(range(len(input_files))):
        file = input_files[i]
        mask_path = os.path.join(mask_dir, f'{file.split("/")[-1].split(".")[0]}.npy')
        
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            mask = mask.astype(float)
            mask = _resize_mask(mask, image_size)
            
            W, H = mask.shape[-1], mask.shape[-2]
            cx, cy = W // 2, H // 2
            mask = center_crop_mask(mask, cx, cy, False)
            
            mask = (mask > 0.5).astype(np.uint8).squeeze(1)
            mask = adaptive_erode_mask(mask.astype(float), 15, 1, 15, 1, 15)
            mask_list.append(mask)
        else:
            mask_list.append(None)


    depth_shape = depth_maps[0].shape[2:]

    for i in range(len(mask_list)):
        if mask_list[i] is None:
            mask_list[i] = np.zeros((0,) + depth_shape, dtype=float)

    return mask_list, input_files

def load_depth_maps(filelist, depth_dir, image_size):
    """Load and process depth maps."""
    depth_maps = []
    input_files = sorted(os.listdir(filelist))
    for i in tqdm(range(len(input_files))):
        file = input_files[i]
        depth_path = os.path.join(depth_dir, f'{file.split("/")[-1].split(".")[0]}.npy')
        depth_map = np.load(depth_path)
        depth_map = np.expand_dims(depth_map, axis=0)
        depth_map = np.expand_dims(depth_map, axis=0)
        depth_map = depth_map.astype(float)
        depth_map = _resize_mask(depth_map, image_size)
        
        W, H = depth_map.shape[-1], depth_map.shape[-2]
        cx, cy = W // 2, H // 2
        depth_map = center_crop_mask(depth_map, cx, cy, False)
        depth_maps.append(depth_map)
    return depth_maps

def load_clip_features(filelist, mask_dir):
    """Load CLIP features for each image."""
    clip_features_list = []
    input_files = sorted(os.listdir(filelist))
    for i in tqdm(range(len(input_files))):
        file = input_files[i]
        clip_path = os.path.join(mask_dir, f'{file.split("/")[-1].split(".")[0]}_clip_features.npy')
        if os.path.exists(clip_path):
            clip_features = np.load(clip_path)
            clip_features_list.append(clip_features)
        else:
            print(f"Warning: CLIP features not found at {clip_path}")
            clip_features_list.append([])
    return clip_features_list

def load_labels(json_dir, filelist, mode='default'):
    """Load JSON labels file based on filelist.
    
    Args:
        json_dir: Directory containing the JSON file
        filelist: Path to the image directory
        mode: 'ground', 'window', or 'default'
    
    Returns:
        labels_by_frame: List of labels for each frame in filelist
    """
    if mode == 'ground':
        json_filename = 'labels_by_frame_ground.json'
    elif mode == 'window':
        json_filename = 'labels_by_frame_window.json'
    else:
        json_filename = 'labels_by_frame.json'
    
    json_path = os.path.join(json_dir, json_filename)
    
    input_files = sorted(os.listdir(filelist))
    labels_by_frame = []
    
    data = None
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
    else:
        labels_by_frame = [[] for _ in input_files]
        return labels_by_frame

    if isinstance(data, dict):
        for file in input_files:
            if file in data:
                labels_by_frame.append(data[file])
            else:
                labels_by_frame.append([])
    else:
        labels_by_frame = data
    
    return labels_by_frame

def run_inference_and_align(images, model, device, lit_module, min_conf_thr):
    """Run inference, estimate camera poses, and align points to global coords."""
    output_dict, _ = inference(
        images,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )

    poses_c2w_batch, _ = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )

    lit_module.align_local_pts3d_to_global(preds=output_dict['preds'], views=output_dict['views'], min_conf_thr_percentile=min_conf_thr)
    
    return output_dict

def process_objects(unique_objects, input_files, mask_list, labels_by_frame, 
                   output_dict, scale, min_conf_thr, clip_features_list):
    """Extract 3D points, CLIP features, and colors for all objects in one pass."""
    obj_pts3d, clip_features, obj_colors = [], [], []
    
    for obj in unique_objects:
        mask_list_obj = build_mask_list_for_object(obj, input_files, mask_list, labels_by_frame)
        
        pts3d = get_3D_obj_from_scene(
            output_dict['preds'], output_dict['views'], mask_list_obj, scale, min_conf_thr_percentile=min_conf_thr
        )
        obj_pts3d.append(np.concatenate(pts3d, axis=0) if len(pts3d) > 0 else np.zeros((0, 3)))
        
        # Only extract CLIP features if clip_features_list is provided (default mode)
        if clip_features_list is not None:
            clip_features.append(aggregate_clip_features_for_object(
                clip_features_list, mask_list_obj, labels_by_frame, obj
            ))
        
        obj_colors.append(extract_object_colors_from_views(
            output_dict['preds'], output_dict['views'], mask_list_obj, min_conf_thr
        ))
    
    obj_scene_colors = np.concatenate(obj_colors, axis=0) if obj_colors and any(c.shape[0] > 0 for c in obj_colors) else None
    return obj_pts3d, clip_features, obj_scene_colors

def save_object_scene_ply(reconstruction_obj_dir, obj_scene_points, obj_scene_colors, mode='default'):
    """Save object scene point cloud as PLY file."""
    os.makedirs(reconstruction_obj_dir, exist_ok=True)
    
    # Determine filename based on mode
    if mode == 'window':
        ply_filename = 'window_obj_scene.ply'
    elif mode == 'ground':
        ply_filename = 'ground_obj_scene.ply'
    else:
        ply_filename = 'obj_scene.ply'
    
    obj_scene_ply_path = os.path.join(reconstruction_obj_dir, ply_filename)
    if obj_scene_points.shape[0] > 0 and obj_scene_colors is not None and obj_scene_colors.shape[0] == obj_scene_points.shape[0]:
        obj_scene_pc = trimesh.PointCloud(vertices=obj_scene_points, colors=(obj_scene_colors * 255).astype(np.uint8))
        obj_scene_pc.export(obj_scene_ply_path)
        print(f"Saved obj_scene ply: {obj_scene_ply_path}, points: {obj_scene_points.shape[0]}")

def save_object_results(bbox_output_dir, obj_pts3d, clip_features, mode='default'):
    """Save object point clouds and CLIP features."""
    os.makedirs(bbox_output_dir, exist_ok=True)
    
    # Determine filename based on mode
    if mode == 'window':
        pts3d_filename = 'window_pts3d.npz'
    elif mode == 'ground':
        pts3d_filename = 'ground_pts3d.npz'
    else:
        pts3d_filename = 'obj_pts3d.npz'
    
    pts3d_to_save = obj_pts3d if obj_pts3d else [np.zeros((0, 3))]
    np.savez(os.path.join(bbox_output_dir, pts3d_filename), *pts3d_to_save)
    
    if len(clip_features) > 0:
        assert len(obj_pts3d) == len(clip_features), (
            f"Length mismatch: obj_pts3d has {len(obj_pts3d)} objects, clip_features has {len(clip_features)} objects"
        )
        all_clip_features = np.stack(clip_features, axis=0)
        clip_features_path = os.path.join(bbox_output_dir, 'clip_feature.npy')
        np.save(clip_features_path, all_clip_features)
        print(f"Saved CLIP features: {clip_features_path}, shape: {all_clip_features.shape}")

def get_reconstructed_scene(filelist, image_size, model, device, output_dir, min_conf_thr, base_dir, scene_dir, mode='default'):
    filelist = filelist[0]

    filelist_path = Path(filelist)
    scene_name = filelist_path.name

    if mode == 'window':
        mask_subdir = 'mask_window'
    elif mode == 'ground':
        mask_subdir = 'mask_ground'
    else:
        mask_subdir = 'mask'
    
    mask_dir = os.path.join(base_dir, 'scene_data', mask_subdir, scene_dir, scene_name)
    depth_dir = os.path.join(base_dir, 'scene_data', 'depth', scene_dir, scene_name)
    json_dir = os.path.join(base_dir, 'scene_data', 'json_files', scene_dir, scene_name)
    bbox_dir = os.path.join(base_dir, 'scene_data', 'bbox_3d', scene_dir, scene_name)
    reconstruction_obj_dir = os.path.join(base_dir, 'scene_data', 'obj_reconstruction', scene_dir, scene_name)

    images = load_images(filelist, size=image_size, verbose=True)
    depth_maps = load_depth_maps(filelist, depth_dir, image_size)

    mask_list, input_files = load_masks(filelist, mask_dir, image_size, depth_maps)
    labels_by_frame = load_labels(json_dir, filelist, mode=mode)
    clip_features_list = load_clip_features(filelist, mask_dir) if mode == 'default' else None
    
    assert len(labels_by_frame) == len(images), f"labels_by_frame length ({len(labels_by_frame)}) and images length ({len(images)}) mismatch"
    assert len(mask_list) == len(images), f"mask_list length ({len(mask_list)}) and images length ({len(images)}) mismatch"
    if clip_features_list is not None:
        assert len(clip_features_list) == len(images), f"clip_features_list length ({len(clip_features_list)}) and images length ({len(images)}) mismatch"


    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()

    output_dict = run_inference_and_align(images, model, device, lit_module, min_conf_thr)
    
    colors, scale = plot_3d_points_with_colors(
        output_dict['preds'], 
        output_dict['views'], 
        depth_maps=depth_maps,
        min_conf_thr_percentile=min_conf_thr,
        export_ply_path=str(output_dir / 'scene.ply')
    )

    unique_objects = list(set(num for sublist in labels_by_frame if sublist for num in sublist))
    unique_objects.sort() 

    obj_pts3d, clip_features, obj_scene_colors = process_objects(
        unique_objects, input_files, mask_list, labels_by_frame,
        output_dict, scale, min_conf_thr, clip_features_list
    )

    obj_scene_points = np.concatenate(obj_pts3d, axis=0) if len(obj_pts3d) > 0 else np.zeros((0, 3))

    save_object_scene_ply(reconstruction_obj_dir, obj_scene_points, obj_scene_colors, mode=mode)
    save_object_results(bbox_dir, obj_pts3d, clip_features, mode=mode)

    return output_dict, str(output_dir / 'scene.ply')

def main_cli(model, device, image_size, input_files, output_dir, 
            min_conf_thr=3.0, base_dir=None, scene_dir=None, mode='default'):
    """
    Main function for command line interface
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input files
    if isinstance(input_files, str):
        filelist = [input_files]
    elif os.path.isdir(input_files):
        filelist = [os.path.join(input_files, f) for f in os.listdir(input_files)]
    else:
        filelist = input_files

    get_reconstructed_scene(filelist, image_size, model, device, output_dir, min_conf_thr, base_dir, scene_dir, mode)
    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
    model = model.to(args.device)

    scene_dir_path = Path(args.scene_dir)
    base_dir = str(scene_dir_path.parent.parent.parent)
    scene_dir = scene_dir_path.name
    
    print(f"Extracted BASE_DIR: {base_dir}")
    print(f"Extracted SCENE_DIR: {scene_dir}")

    # Get all scene subdirectories
    scene_dir_path = Path(args.scene_dir)
    scene_subdirs = [d for d in scene_dir_path.iterdir() if d.is_dir()]
    scene_subdirs.sort()

    for scene_subdir in scene_subdirs:
        scene_name = scene_subdir.name
        print(f"Processing scene: {scene_name}")
        
        input_path = str(scene_subdir)
        output_dir = os.path.join(args.output_dir, scene_name)
        
        with tempfile.TemporaryDirectory(suffix='fast3r_cli'):
            main_cli(
                model,
                args.device,
                args.image_size,
                input_path,
                output_dir,
                min_conf_thr=args.min_conf_thr,
                base_dir=base_dir,
                scene_dir=scene_dir,
                mode=args.mode
            )