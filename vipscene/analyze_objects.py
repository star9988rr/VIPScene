#!/usr/bin/env python3
import os
import json
import base64
import time
import argparse
from pathlib import Path
import openai
import imageio
from constants import API_KEY

openai.api_key = API_KEY

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, max_tokens=1000):
    """Analyze image with GPT-4o to identify objects."""
    try:
        base64_image = encode_image(image_path)
        
        prompt = """Please list all the objects in this room in detail, excluding elements like windows, floors, walls, and ceilings. Focus only on specific physical objects, and make sure to include smaller items as well. List only the general categories (no quantities or locations), and present the result in the format:
['sofa', 'throw pillow', ...]"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse the response as a list
        try:
            if content.startswith('[') and content.endswith(']'):
                objects_list = eval(content)
            else:
                # Try to find the list in the response
                import re
                list_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                if list_match:
                    objects_list = eval('[' + list_match.group(1) + ']')
                else:
                    objects_list = [content]
            
            return objects_list
        except:
            # If parsing fails, return the raw content as a single item
            return [content]
            
    except Exception as e:
        print(f"Error analyzing {image_path}: {str(e)}")
        return []

def extract_frames_from_video(video_path, output_dir, n_frame):
    """Extract frames from video and save them."""
    frames = imageio.v3.imread(video_path)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    for i in range(n_frame):
        img_id = int(frames.shape[0] * i // n_frame)
        frame = frames[img_id]
        
        frame_filename = os.path.join(frame_dir, f"frame_{i:06d}.jpg")
        imageio.imwrite(frame_filename, frame)
    
    return frame_dir

def process_scene(scene_dir, prompt, max_tokens=1000, delay=1.0):
    """Process all frames in a scene directory and merge results."""
    all_objects = set()
    frame_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('.jpg')])
    
    for frame_file in frame_files:
        frame_path = os.path.join(scene_dir, frame_file)
        objects = analyze_image(frame_path, max_tokens)
        all_objects.update(objects)
        
        # Add a small delay to avoid rate limiting
        time.sleep(delay)
    
    result = {
        "prompt": prompt,
        "objects": sorted(list(all_objects))
    }
    
    return [result]

def main():
    """Main function to extract frames from video and analyze objects."""
    parser = argparse.ArgumentParser(description='Extract frames from video and analyze objects using GPT-4o')
    parser.add_argument('--video_path', '-v', 
                       help='Path to input video folder containing video files')
    parser.add_argument('--output_dir', '-o',
                       help='Output directory for extracted frames and results')
    parser.add_argument('--prompt_json', '-p', type=str,
                       help='JSON file containing prompts for each video')
    parser.add_argument('--n_frame', '-n', type=int, default=10,
                       help='Number of frames to extract from video')
    
    args = parser.parse_args()
    
    if not args.video_path:
        print("Error: --video_path is required!")
        return
    
    if not args.output_dir:
        print("Error: --output_dir is required!")
        return
    
    if not args.prompt_json:
        print("Error: --prompt_json is required!")
        return
    
    if not os.path.exists(args.prompt_json):
        print(f"Error: Prompt JSON file '{args.prompt_json}' does not exist!")
        return
    
    # Load prompts from JSON file
    try:
        with open(args.prompt_json, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)
    except Exception as e:
        print(f"Error loading prompt JSON file: {str(e)}")
        return
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video folder '{args.video_path}' does not exist!")
        return
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    video_files = []
    for file in os.listdir(args.video_path):
        file_path = os.path.join(args.video_path, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file.lower())
            if ext in video_extensions:
                video_files.append(file_path)
    
    if not video_files:
        print(f"Error: No video files found in '{args.video_path}'!")
        return
    
    video_files.sort()
    print(f"Found {len(video_files)} video file(s) to process: {[os.path.basename(v) for v in video_files]}\n")
    
    all_results = []
    
    for video_idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing video {video_idx + 1}/{len(video_files)}: {video_name}")
        print(f"{'='*60}\n")
        
        try:
            # Get prompt for this video
            if video_name not in prompts_dict:
                print(f"Warning: No prompt found for video '{video_name}', skipping...")
                continue
            
            prompt = prompts_dict[video_name]
            
            frame_dir = extract_frames_from_video(
                video_path, 
                args.output_dir, 
                args.n_frame
            )
            
            scene_results = process_scene(frame_dir, prompt)
            print(f"Completed video {video_name}: {len(scene_results)} merged result(s) created\n")
            
            all_results.extend(scene_results)
            
        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results to a single JSON file
    output_file = os.path.join(args.output_dir, 'object_analysis_results.json')
    print(f"\nSaving all results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"All videos processed!")
    print(f"Total scenes: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
