from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.utils.image import load_images
import os
import numpy as np
import torch
import json
import argparse
from collections import Counter
from typing import Tuple, Dict

def load_model(model_ckpt: str, device: str):
    return AsymmetricMASt3R.from_pretrained(model_ckpt).to(device)


def list_scenes(image_data_root: str):
    """List all scene directories."""
    if not os.path.isdir(image_data_root):
        return []
    return [os.path.join(image_data_root, x) 
            for x in os.listdir(image_data_root) 
            if os.path.isdir(os.path.join(image_data_root, x))]


def list_scene_frames(image_data_root: str, scene: str, mask_root: str):
    """List all frame files in a scene that have corresponding mask files, sorted."""
    scene_dir = os.path.join(image_data_root, scene)
    if not os.path.isdir(scene_dir):
        return []
    mask_dir = os.path.join(mask_root, scene)
    return sorted([f for f in os.listdir(scene_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and os.path.exists(os.path.join(mask_dir, f"{f.split('.')[0]}.npy"))])


def init_first_frame(scene: str, image_data_root: str, mask_root: str):
    """Initialize first frame with labels 0, 1, 2, ..."""
    files = list_scene_frames(image_data_root, scene, mask_root)
    if not files:
        return [], {}, None
    
    first_file = files[0].split('.')[0]
    first_mask = np.load(os.path.join(mask_root, scene, f"{first_file}.npy"))
    current_labels = list(range(first_mask.shape[0]))
    max_label_so_far = max(current_labels) if current_labels else -1
    labels_by_frame = {files[0]: current_labels}
    
    print(f"[SCENE {scene}] First frame labels: {current_labels}")
    
    return files, labels_by_frame, max_label_so_far


def resize_and_binarize_mask(mask_np: np.ndarray, target_h: int, target_w: int, device: str) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask_np).float().to(device)
    mask_tensor = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(1),
        size=(target_h, target_w),
        mode='nearest'
    ).squeeze(1).cpu().numpy()
    return (mask_tensor > 0.5).astype(np.uint8)


def build_assignment(mask_binary: np.ndarray) -> np.ndarray:
    n, h, w = mask_binary.shape
    mask_flat = mask_binary.reshape(n, -1)
    assignment = np.argmax(mask_flat, axis=0).reshape(h, w)
    assignment[~mask_binary.any(axis=0).squeeze()] = -1
    return assignment


def is_within_boundary(matches: np.ndarray, width: int, height: int, margin: int = 3) -> np.ndarray:
    """Check if matches are within image boundaries with margin."""
    return ((matches[:, 0] >= margin) & (matches[:, 0] < width - margin) &
            (matches[:, 1] >= margin) & (matches[:, 1] < height - margin))


def get_most_common_valid(values: np.ndarray, counts: np.ndarray) -> int:
    """Get the most common value that is not -1. Returns -1 if no valid value exists."""
    valid_mask = values != -1
    if not valid_mask.any():
        return -1
    valid_values = values[valid_mask]
    valid_counts = counts[valid_mask]
    return valid_values[np.argmax(valid_counts)]


def filter_valid_coords(matches_im1: np.ndarray, matches_im0: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Filter matches to ensure coordinates are within image boundaries."""
    coords_im1 = matches_im1.astype(int)
    coords_im0 = matches_im0.astype(int)
    valid_mask = ((coords_im1[:, 0] >= 0) & (coords_im1[:, 0] < w) &
                  (coords_im1[:, 1] >= 0) & (coords_im1[:, 1] < h) &
                  (coords_im0[:, 0] >= 0) & (coords_im0[:, 0] < w) &
                  (coords_im0[:, 1] >= 0) & (coords_im0[:, 1] < h))
    return coords_im1[valid_mask], coords_im0[valid_mask]


def process_scene(scene: str,
                 files: list,
                 image_data_root: str,
                 mask_root: str,
                 model,
                 device: str,
                 labels_by_frame: Dict,
                 max_label_so_far: int):
    """
    Process a scene by propagating segmentation labels across frames using feature matching.
    For each frame, matches it with all previous frames to establish label correspondences.
    """
    # Process each frame sequentially (starting from frame 1, frame 0 is the reference)
    for i in range(1, len(files)):
        file_2 = os.path.join(image_data_root, scene, files[i])
        match_list = []  # Store matches with all previous reference frames

        # Match current frame with all previous frames
        for j in range(0, i):
            file_1 = os.path.join(image_data_root, scene, files[j])

            # Extract feature descriptors using MASt3R model
            images = load_images([file_1, file_2], size=512)
            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

            # Find reciprocal nearest neighbor matches between frames
            matches_im0, matches_im1 = fast_reciprocal_NNs(
                desc1,
                desc2,
                subsample_or_initxy1=8,
                device=device,
                dist='dot',
                block_size=2**13
            )

            # Filter matches near image boundaries (less reliable)
            H0, W0 = view1['true_shape'][0]
            H1, W1 = view2['true_shape'][0]
            valid_matches_im0 = is_within_boundary(matches_im0, int(W0), int(H0))
            valid_matches_im1 = is_within_boundary(matches_im1, int(W1), int(H1))

            valid_matches = valid_matches_im0 & valid_matches_im1
            matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

            # Load and process segmentation masks
            file_1_base = files[j].split('.')[0]
            file_2_base = files[i].split('.')[0]
            ori_mask1 = np.load(os.path.join(mask_root, scene, f"{file_1_base}.npy")).squeeze(axis=1)
            ori_mask2 = np.load(os.path.join(mask_root, scene, f"{file_2_base}.npy")).squeeze(axis=1)

            n1 = ori_mask1.shape[0]
            n2 = ori_mask2.shape[0]

            # Resize masks to match feature extraction resolution
            mask1 = resize_and_binarize_mask(ori_mask1, H0, W0, device)
            mask2 = resize_and_binarize_mask(ori_mask2, H0, W0, device)

            h, w = mask1.shape[1], mask1.shape[2]

            # Build assignment maps: each pixel -> mask index
            mask1_assignment = build_assignment(mask1)
            mask2_assignment = build_assignment(mask2)

            # Filter matches to ensure coordinates are within mask dimensions
            coords_im1, coords_im0 = filter_valid_coords(matches_im1, matches_im0, h, w)

            if coords_im1.size == 0:
                continue

            # Extract mask indices at matched point locations
            y1, x1 = coords_im1[:, 1], coords_im1[:, 0]
            y0, x0 = coords_im0[:, 1], coords_im0[:, 0]

            mask2_indices = mask2_assignment[y1, x1]  # Current frame mask indices at match points
            mask1_labels = mask1_assignment[y0, x0]   # Reference frame mask indices at match points
            match_list.append((mask2_indices, mask1_labels, j))  # Store: (current_mask_idx, ref_mask_idx, ref_frame_idx)

        # Handle case where no valid matches found with any previous frame
        if not match_list:
            current_labels = list(range(max_label_so_far + 1, max_label_so_far + 1 + n2))
            max_label_so_far += n2
            labels_by_frame[files[i]] = current_labels
            print(f"[SCENE {scene}] Frame {i} labels: {current_labels} (no matches, new labels)")
            continue

        # Assign labels to each mask in current frame using voting mechanism
        best_labels = np.full(n2, -1, dtype=int)

        for m2_idx in range(n2):
            potential_labels = []  # Collect label suggestions from all reference frames
            
            # For each reference frame, find the most common corresponding label
            for mask2_indices, mask1_labels, j in match_list:
                # Check if current mask appears in this reference frame's matches
                if m2_idx not in mask2_indices:
                    continue
                
                # Get corresponding reference frame mask indices for current mask
                labels = mask1_labels[mask2_indices == m2_idx]
                values, counts = np.unique(labels, return_counts=True)
                
                # Get most common valid (non -1) label
                value = get_most_common_valid(values, counts)
                if value != -1:
                    potential_labels.append(labels_by_frame[files[j]][value])

            # Vote among all reference frames: assign most common suggested label
            if potential_labels:
                best_labels[m2_idx] = Counter(potential_labels).most_common(1)[0][0]
            else:
                # No matches: assign new unique label
                max_label_so_far += 1
                best_labels[m2_idx] = max_label_so_far

        current_labels = best_labels.tolist()
        labels_by_frame[files[i]] = current_labels
        print(f"[SCENE {scene}] Frame {i} labels: {current_labels}")

    return labels_by_frame, max_label_so_far


def save_scene_outputs(scene: str, labels_json_root: str, labels_by_frame: Dict, task: str):
    """Save label propagation results to JSON file."""
    os.makedirs(os.path.join(labels_json_root, scene), exist_ok=True)
    output_path = os.path.join(labels_json_root, scene, f'labels_by_frame_{task}.json')
    with open(output_path, 'w') as f:
        json.dump(labels_by_frame, f)


def resolve_paths(scene_dir: str, task: str):
    """Resolve paths for image data, masks, and JSON files."""
    subset_name = os.path.basename(os.path.normpath(scene_dir))
    image_data_root = scene_dir
    dataset_root = os.path.dirname(os.path.dirname(image_data_root))
    mask_root = os.path.join(dataset_root, f'mask_{task}', subset_name)
    labels_json_root = os.path.join(dataset_root, 'json_files', subset_name)
    scenes = list_scenes(image_data_root)
    return image_data_root, mask_root, labels_json_root, scenes


def parse_args():
    parser = argparse.ArgumentParser(description="MASt3R label propagation for window/ground masks")
    parser.add_argument("--scene_dir", required=True, help='scene directory path')
    parser.add_argument("--model_ckpt", required=True, help='path to model checkpoint or repo id')
    parser.add_argument("--task", choices=["window", "ground"], default="window", help='which mask type to use')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    scene_dir = args.scene_dir
    model_ckpt = args.model_ckpt
    task = args.task
    image_data_root, mask_root, labels_json_root, scenes = resolve_paths(scene_dir, task)

    device = 'cuda'
    model = load_model(model_ckpt, device)

    for path in sorted(scenes):
        scene = os.path.basename(path).split('.')[0]

        files, labels_by_frame, max_label_so_far = init_first_frame(
            scene, image_data_root, mask_root
        )
        
        if not files:
            continue

        labels_by_frame, max_label_so_far = process_scene(
            scene=scene,
            files=files,
            image_data_root=image_data_root,
            mask_root=mask_root,
            model=model,
            device=device,
            labels_by_frame=labels_by_frame,
            max_label_so_far=max_label_so_far
        )
        
        save_scene_outputs(scene, labels_json_root, labels_by_frame, task)

