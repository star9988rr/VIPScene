import ast
import json
import os
import traceback
from argparse import ArgumentParser

import compress_json
from tqdm import tqdm

from vipscene.constants import HOLODECK_BASE_DATA_DIR, OBJATHOR_ASSETS_DIR
from vipscene.generation.vipscene import VIPScene


def str2bool(v: str):
    v = v.lower().strip()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"{v} cannot be converted to a bool")


def generate_single_scene(args):

    scene = None
    if args.original_scene is not None:
        print(f"Loading original scene from {args.original_scene}.")
        try:
            scene = compress_json.load(args.original_scene)
        except:
            print(
                f"[ERROR] Could not load original scene from given path {args.original_scene}."
            )
            raise
    else:
        # Try to load existing scene if scene_name is provided
        if hasattr(args, 'scene_name') and args.scene_name:
            scene_name = args.scene_name
            key = getattr(args, 'key', None)
            path = os.path.join(
                args.save_dir, f"scenes/{scene_name}/{key}.json"
            )
            if os.path.exists(path):
                print(f"Loading existing scene from {path}.")
                try:
                    scene = compress_json.load(path)
                except:
                    print(
                        f"[ERROR] The path {path} exists but could not be loaded. Please delete"
                        f" this file and try again."
                    )
                    raise

    if scene is None:
        print("Generating from an empty scene.")
        scene = args.model.get_empty_scene()

    try:
        _, save_dir = args.model.generate_scene(
            scene=scene,
            plan_json=args.plan_json,
            query=args.query,
            save_dir=args.save_dir,
            used_assets=args.used_assets,
            generate_image=ast.literal_eval(args.generate_image),
            generate_video=ast.literal_eval(args.generate_video),
            add_ceiling=ast.literal_eval(args.add_ceiling),
            add_time=ast.literal_eval(args.add_time),
            use_constraint=ast.literal_eval(args.use_constraint),
            use_milp=ast.literal_eval(args.use_milp),
            random_selection=ast.literal_eval(args.random_selection),
        )
    except:
        print(
            f"[ERROR] Could not generate scene from {args.query}. Traceback:\n{traceback.format_exc()}"
        )
        return

    print(
        f"Generation complete for {args.query}. Scene saved and any other data saved to {save_dir}."
    )


def generate_multi_scenes(args):
    with open(args.query_file, "r") as f:
        query_data = json.load(f)
    
    scene_name = args.query_file.split('/')[-1].split('.')[0]
    base_save_dir = args.save_dir
    
    sorted_keys = sorted(query_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    for key in tqdm(sorted_keys, desc=f"Processing {scene_name}"):
        query = query_data[key]
        args.query = query
        args.scene_name = scene_name
        args.key = key
        
        plan_json_path = os.path.join(args.input_dir, key, "scene_with_assets.json")
        args.plan_json = plan_json_path if os.path.exists(plan_json_path) else None
        
        args.save_dir = os.path.join(base_save_dir, scene_name, key)
        os.makedirs(args.save_dir, exist_ok=True)

        generate_single_scene(args)


def generate_variants(args):
    try:
        original_scene = compress_json.load(args.original_scene)
    except:
        raise Exception(f"Could not load original scene from {args.original_scene}.")

    try:
        args.model.generate_variants(
            query=args.query,
            original_scene=original_scene,
            save_dir=args.save_dir,
            number_of_variants=int(args.number_of_variants),
            used_assets=args.used_assets,
        )
    except:
        print(f"Could not generate variants from {args.query}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Mode to run in (generate_single_scene, generate_multi_scenes or generate_variants).",
        default="generate_single_scene",
    )
    parser.add_argument(
        "--input_dir", help="Directory containing numbered folders with scene_with_assets.json files.", default=None
    )
    parser.add_argument(
        "--query", help="Query to generate scene from.", default="a living room"
    )
    parser.add_argument(
        "--query_file", help="JSON file with numbered keys containing queries."
    )
    parser.add_argument(
        "--number_of_variants", help="Number of variants to generate.", default=5
    )
    parser.add_argument(
        "--original_scene",
        help="Original scene to generate variants from or to load as base scene.",
        default=None,
    )
    parser.add_argument(
        "--openai_api_key",
        help="OpenAI API key. If none given, will attempt to read this from the OPENAI_API_KEY env variable.",
        default=None,
    )
    parser.add_argument(
        "--openai_org",
        help="OpenAI ORG string. If none given, will attempt to read this from the OPENAI_ORG env variable.",
        default=None,
    )
    parser.add_argument(
        "--save_dir", help="Directory to save scene to.", default="./data/scenes"
    )
    parser.add_argument(
        "--generate_image",
        help="Whether to generate an image of the scene.",
        default="True",
    )
    parser.add_argument(
        "--generate_video",
        help="Whether to generate a video of the scene.",
        default="False",
    )
    parser.add_argument(
        "--add_ceiling", help="Whether to add a ceiling to the scene.", default="False"
    )
    parser.add_argument(
        "--add_time", help="Whether to add the time to the scene name.", default="True"
    )
    parser.add_argument(
        "--use_constraint", help="Whether to use constraints.", default="True"
    )
    parser.add_argument(
        "--use_milp",
        help="Whether to use mixed integer linear programming for the constraint satisfaction solver.",
        default="False",
    )
    parser.add_argument(
        "--random_selection",
        help="Whether to more random object selection, set to False will be more precise, True will be more diverse",
        default="False",
    )
    parser.add_argument(
        "--used_assets",
        help="a list of assets which we want to exclude from the scene",
        default=[],
    )
    parser.add_argument(
        "--single_room",
        help="Whether to generate a single room scene.",
        default="False",
    )

    args = parser.parse_args()

    if args.openai_api_key is None:
        args.openai_api_key = os.environ.get("OPENAI_API_KEY")

    if args.openai_org is None:
        args.openai_org = os.environ.get("OPENAI_ORG")

    args.model = VIPScene(
        openai_api_key=args.openai_api_key,
        openai_org=args.openai_org,
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=ast.literal_eval(args.single_room),
    )

    if args.used_assets != [] and args.used_assets.endswith(".txt"):
        with open(args.used_assets, "r") as f:
            args.used_assets = f.readlines()
            args.used_assets = [asset.strip() for asset in args.used_assets]
    else:
        args.used_assets = []

    if args.mode == "generate_single_scene":
        generate_single_scene(args)

    elif args.mode == "generate_multi_scenes":
        generate_multi_scenes(args)

    elif args.mode == "generate_variants":
        generate_variants(args)

    else:
        raise Exception(f"Mode {args.mode} not supported.")
