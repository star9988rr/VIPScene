# Usage Pipeline

This document outlines the pipeline for analyzing video data, processing it with various perception models, and generating the final 3D scene using VIPScene.


### 0. Data Preparation
Before running the pipeline, ensure you have prepared your data:

1. **Video Files**: Place your video files (e.g., `0.mp4`, `1.mp4`) in the `video/$SCENE_DIR` directory.
2. **Scene Descriptions**: Create a JSON file at `prompt_files/$SCENE_DIR.json` containing scene descriptions.

The JSON file should map video filenames to their corresponding scene descriptions. For example:

```json
{
  "0": "A comfortable living room with a sofa, a coffee table, a TV stand with a television, bookshelves, a floor lamp, a side table, and a decorative plant.",
  "1": "A cozy bedroom with a queen bed, a desk and chair, a wardrobe, nightstands on each side of the bed, a bedside lamp"
}
```

### 1. Analyze Video and Extract Labels
First, analyze the objects in the video to extract their labels.
```bash
# Define your base and scene directories
export BASE_DIR="/path/to/vipscene"
export SCENE_DIR='example_scene'
export GPU=0

python vipscene/analyze_objects.py \
    --video_path "$BASE_DIR/scene_data/video/$SCENE_DIR" \
    --output_dir "$BASE_DIR/scene_data/image_data/$SCENE_DIR" \
    --prompt_json "$BASE_DIR/scene_data/prompt_files/$SCENE_DIR.json" \
    --n_frame 10
```

### 2. Depth Estimation
Copy `external_scripts/unidepth/run_unidepth.py` to the UniDepth codebase directory and run it to estimate video depth.

```bash
CUDA_VISIBLE_DEVICES=$GPU python run_unidepth.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR \
    --unidepth_path /path/to/UniDepth
```

### 3. Object Detection
Copy `external_scripts/grounded_sam/run_grounded_sam.py` to the Grounded-SAM codebase directory. Run the following commands to robustly detect objects in the video frames.

```bash
# General detection
CUDA_VISIBLE_DEVICES=$GPU python run_grounded_sam.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR

# Ground detection
CUDA_VISIBLE_DEVICES=$GPU python run_grounded_sam.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR \
    --mode ground

# Window detection
CUDA_VISIBLE_DEVICES=$GPU python run_grounded_sam.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR \
    --mode window
```

### 4. Correspondence Matching
Copy `external_scripts/mast3r/run_mast3r.py` and `external_scripts/mast3r/run_mast3r_window_ground.py` to the Mast3R codebase directory. Run them to obtain correspondences between frames.

```bash
# General correspondence
CUDA_VISIBLE_DEVICES=$GPU python run_mast3r.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR \
    --model_ckpt /path/to/mast3r/checkpoints

# Window correspondence
CUDA_VISIBLE_DEVICES=$GPU python run_mast3r_window_ground.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR \
    --model_ckpt /path/to/mast3r/checkpoints \
    --task window

# Ground correspondence
CUDA_VISIBLE_DEVICES=$GPU python run_mast3r_window_ground.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR \
    --model_ckpt /path/to/mast3r/checkpoints \
    --task ground
```

### 5. Scene Geometry Reconstruction
Copy `external_scripts/fast3r/run_fast3r.py` to the Fast3R codebase directory. Run it to reconstruct scene geometry and each object geometry.

```bash
CUDA_VISIBLE_DEVICES=$GPU python run_fast3r.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR
```

### 6. Extract Object Information
Extract information for each object in the scene.

```bash
python vipscene/extract_scene_info.py \
    --scene_dir $BASE_DIR/scene_data/image_data/$SCENE_DIR
```

### 7. Retrieve 3D Assets
Retrieve 3D assets based on the analyzed data.

```bash
python vipscene/obj_retrieval.py \
    --input_dir "$BASE_DIR/scene_data/image_data/$SCENE_DIR"
```

### 8. Scene Optimization
Finally, optimize to generate the final 3D scene.

```bash
python vipscene/main.py \
    --mode "generate_multi_scenes" \
    --query_file "$BASE_DIR/scene_data/prompt_files/$SCENE_DIR.json" \
    --input_dir "$BASE_DIR/scene_data/json_files/$SCENE_DIR" \
    --generate_image True
```

