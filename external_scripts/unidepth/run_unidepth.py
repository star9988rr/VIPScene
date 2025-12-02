import torch
import numpy as np
import os
import argparse
import matplotlib as mpl
import imageio
from pathlib import Path
from PIL import Image
from tqdm import tqdm

torch.backends.cudnn.enabled = False

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(directory):
    directory_path = Path(directory)
    return [f for f in directory_path.iterdir() 
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]


def process_image(model, filename, depth_dir, intrinsics_dir):
    file_name = Path(filename).stem
    
    rgb = torch.from_numpy(np.array(Image.open(filename).convert("RGB"))).permute(2, 0, 1)
    predictions = model.infer(rgb)
    depth = predictions["depth"]
    intrinsics = predictions["intrinsics"]
    
    depth_np = depth.cpu().numpy().squeeze(0).squeeze(0)
    np.save(Path(depth_dir) / file_name, depth_np)
    
    intrinsics_np = intrinsics.cpu().numpy().reshape(3, 3)
    np.save(Path(intrinsics_dir) / file_name, intrinsics_np)


def main():
    parser = argparse.ArgumentParser(description="UniDepth inference script")
    parser.add_argument("--scene_dir", required=True, help="scene directory path")
    parser.add_argument("--unidepth_path", required=True, help="path to UniDepth repository")
    args = parser.parse_args()
    
    version = "v1"
    backbone = "ViTL14"
    
    model = torch.hub.load(
        args.unidepth_path, "UniDepth", 
        version=version, backbone=backbone, 
        pretrained=True, trust_repo=True, 
        force_reload=True, source='local'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    scene_dir_path = Path(args.scene_dir)
    scene_dir = scene_dir_path.name
    data_folder = scene_dir_path.parent.parent
    image_data_path = data_folder / "image_data" / scene_dir
    
    if not image_data_path.exists():
        raise ValueError(f"Image data path does not exist: {image_data_path}")
    
    scene_list = [d for d in image_data_path.iterdir() 
                  if d.is_dir()]
    scenenames = sorted(scene_list)
    
    for scene in tqdm(scenenames, desc="Processing scenes"):
        scene_name = scene.name
        depth_dir = data_folder / "depth" / scene_dir / scene_name
        intrinsics_dir = data_folder / "intrinsics" / scene_dir / scene_name

        depth_dir.mkdir(parents=True, exist_ok=True)
        intrinsics_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = get_image_files(scene)
        
        for filename in image_files:
            try:
                process_image(model, str(filename), str(depth_dir), str(intrinsics_dir))
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()