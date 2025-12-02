import argparse
import os

import numpy as np
import torch
from PIL import Image
import torchvision

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from segment_anything import build_sam, SamPredictor
import open_clip

from huggingface_hub import hf_hub_download

from tqdm import tqdm
import json


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   



def generate_substrings(input_string):
    input_list = input_string.split()
    result = []
    n = len(input_list)
    for i in range(n):
        for j in range(i, n):
            result.append(' '.join(input_list[i:j+1]))
    return result


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def remove_duplicate_masks(masks, confs):
    kept_indices = [0]
    for i in range(1, len(masks)):
        duplicate_idx = next((k for k in kept_indices if calculate_iou(masks[i], masks[k]) > 0.25), None)
        if duplicate_idx is None:
            kept_indices.append(i)
        else:
            if confs[i] > confs[duplicate_idx]:
                kept_indices.remove(duplicate_idx)
                kept_indices.append(i)
    return kept_indices


def extract_clip_features_from_bboxes(image, bboxes, clip_model, clip_preprocess, device='cuda'):
    clip_features = []
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))
        
        bbox_image = image[y1:y2, x1:x2]
        
        if bbox_image.shape[0] < 10 or bbox_image.shape[1] < 10:
            clip_features.append(np.zeros(768))
            continue
        
        bbox_pil = Image.fromarray(bbox_image)
        
        with torch.no_grad():
            image_tensor = clip_preprocess(bbox_pil).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_features.append(image_features.cpu().numpy().flatten())
    
    return clip_features


def load_models(device, sam_checkpoint, use_clip=True):
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename).to(device)

    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    clip_model = None
    clip_preprocess = None
    if use_clip:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        clip_model = clip_model.to(device)
    
    return groundingdino_model, sam_predictor, clip_model, clip_preprocess


def load_prompts(scene_dir):
    prompt_file_path = os.path.join(scene_dir, 'object_analysis_results.json')
    with open(prompt_file_path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        object_list = item['objects']
        prompt = '. '.join(object_list)
        prompts.append(prompt)
    
    return prompts


def extract_and_match_phrases(phrases, boxes_xyxy, non_zero_idx_list, logits_all, text_prompt):
    boxes_list = []
    conf_list = []
    phrases_list = []
    mask_indices = []

    prompt_segments = set(text_prompt.split('. '))

    for i, phrase in enumerate(phrases):
        if phrase == ' ':
            continue

        non_zero_idx = non_zero_idx_list[i]
        if len(non_zero_idx) == 0:
            continue

        tokens = phrase.split()
        best_conf = -np.inf
        best_substring = None

        for start in range(len(tokens)):
            for end in range(start, len(tokens)):
                substring = ' '.join(tokens[start:end + 1])
                if substring not in prompt_segments:
                    continue
                conf = logits_all[i][non_zero_idx][start:end + 1].mean().item()
                if conf > best_conf:
                    best_conf = conf
                    best_substring = substring

        if best_substring is not None:
            conf_list.append(float(best_conf))
            phrases_list.append(best_substring)
            boxes_list.append(boxes_xyxy[i].cpu().numpy())
            mask_indices.append(i)

    return boxes_list, conf_list, phrases_list, mask_indices


def process_image(filename, text_prompt, groundingdino_model, sam_predictor, device, 
                  box_threshold=0.3, text_threshold=0.25, iou_threshold=0.4):
    image_source, image = load_image(filename)
    
    boxes, logits, phrases, features, encoded_text, non_zero_idx_list, logits_all = predict(
        model=groundingdino_model, 
        image=image,
        caption=text_prompt, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold
    )
    
    if len(boxes) == 0:
        return None

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    nms_idx = torchvision.ops.nms(boxes_xyxy, logits, iou_threshold)
    boxes_xyxy = boxes_xyxy[nms_idx]
    boxes = boxes[nms_idx]
    logits = logits[nms_idx]
    features = features[nms_idx]
    idx_list = nms_idx.cpu().tolist()
    phrases = [phrases[i] for i in idx_list]
    non_zero_idx_list = [non_zero_idx_list[i] for i in idx_list]
    logits_all = [logits_all[i] for i in idx_list]

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    boxes_list, conf_list, phrases_list, mask_indices = extract_and_match_phrases(
        phrases, boxes_xyxy, non_zero_idx_list, logits_all, text_prompt
    )
    
    if len(boxes_list) == 0:
        return None
    
    mask_list = [masks[i].cpu().numpy().astype(bool) for i in mask_indices]
    
    return {
        'image_source': image_source,
        'boxes_list': boxes_list,
        'conf_list': conf_list,
        'phrases_list': phrases_list,
        'mask_list': mask_list,
    }


def save_results(file_name, output_path, result_data, clip_model, clip_preprocess, device, use_clip=True):
    boxes_list = result_data['boxes_list']
    conf_list = result_data['conf_list']
    phrases_list = result_data['phrases_list']
    mask_list = result_data['mask_list']
    image_source = result_data['image_source']
    
    if use_clip and clip_model is not None:
        clip_features_list = extract_clip_features_from_bboxes(
            image_source, boxes_list, clip_model, clip_preprocess, device
        )
    else:
        clip_features_list = None

    indices = remove_duplicate_masks(mask_list, conf_list)

    boxes_list = [boxes_list[i] for i in indices]
    conf_list = [conf_list[i] for i in indices]
    phrases_list = [phrases_list[i] for i in indices]
    mask_list = [mask_list[i] for i in indices]
    
    if clip_features_list is not None:
        clip_features_list = [clip_features_list[i] for i in indices]

    name_dict = {'name': phrases_list}
    json_output_path = os.path.join(output_path, f"{file_name}.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(name_dict, json_file, indent=4)

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, f"{file_name}"), np.array(mask_list))
    
    if clip_features_list is not None:
        np.save(os.path.join(output_path, f"{file_name}_clip_features"), np.array(clip_features_list))
    

def main():
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description="Grounded SAM detection script")
    parser.add_argument("--scene_dir", required=True, help="Directory containing image data and object_analysis_results.json")
    parser.add_argument("--output_dir", required=True, help="Directory for output masks")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--task", choices=["generic", "window", "ground"], default="generic", help="Predefined prompt mode. 'generic' uses object_analysis_results.json; 'window' or 'ground' use fixed prompts")
    args = parser.parse_args()

    if args.task == "window":
        use_clip = False
    elif args.task == "ground":
        use_clip = False
    else:
        use_clip = True
        
    groundingdino_model, sam_predictor, clip_model, clip_preprocess = load_models(args.device, args.sam_checkpoint, use_clip)
    prompts = None
    if args.task == "generic":
        prompts = load_prompts(args.scene_dir)

    video_files = [
        os.path.join(args.scene_dir, x)
        for x in os.listdir(args.scene_dir)
        if os.path.isdir(os.path.join(args.scene_dir, x))
    ]
    video_files.sort()

    for path in video_files:
        scene_name = os.path.basename(path)
        image_data_path = os.path.join(args.scene_dir, scene_name)
        output_path = os.path.join(args.output_dir, scene_name)
        os.makedirs(output_path, exist_ok=True)
        
        file_name_list = os.listdir(image_data_path)
        filenames = [os.path.join(image_data_path, x) for x in file_name_list]
    filenames = sorted(filenames)
        
    for filename in tqdm(filenames):
            file_name = (filename.split('/')[-1])[:-4]

            if args.task == "window":
                text_prompt = "window"
            elif args.task == "ground":
                text_prompt = "floor"
            else:
                text_prompt = prompts[int(scene_name)]

            result_data = process_image(
                filename,
                text_prompt,
                groundingdino_model,
                sam_predictor,
                args.device,
            )
            
            if result_data is None:
                print(f"no box detected in {file_name}")
                continue

            save_results(file_name, output_path, result_data, clip_model, 
                        clip_preprocess, args.device, use_clip)


if __name__ == "__main__":
    main()
