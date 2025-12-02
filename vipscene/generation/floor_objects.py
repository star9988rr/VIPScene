import copy
import json
import math
import multiprocessing
import random
import re
import time

import editdistance
import numpy as np
from langchain import PromptTemplate, OpenAI
from rtree import index
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, box, LineString

import vipscene.generation.prompts as prompts
from vipscene.generation.objaverse_retriever import ObjathorRetriever
from vipscene.generation.utils import get_bbox_dims
from vipscene.generation.utils_optimization import *


class FloorObjectGenerator:
    def __init__(self, object_retriever: ObjathorRetriever, llm: OpenAI):
        self.json_template = {
            "assetId": None,
            "id": None,
            "kinematic": True,
            "position": {},
            "rotation": {},
            "material": None,
            "roomId": None,
        }
        self.llm = llm
        self.object_retriever = object_retriever
        self.database = object_retriever.database

        self.grid_density = 20
        self.add_window = False
        self.size_buffer = 10  # add 10 cm buffer to object size

        self.constraint_type = "llm"
        self.use_milp = False
        self.multiprocessing = False

    def generate_objects(self, scene, plan_json, use_constraint=True):
        rooms = scene["rooms"]
        doors = scene["doors"]
        windows = scene["windows"]
        open_walls = scene["open_walls"]
        selected_objects = scene["selected_objects"]
        results = []

        packed_args = [
            (plan_json, room, doors, windows, open_walls, selected_objects, use_constraint)
            for room in rooms
        ]
        if self.multiprocessing:
            pool = multiprocessing.Pool(processes=4)
            all_placements = pool.map(self.generate_objects_per_room, packed_args)
            pool.close()
            pool.join()
        else:
            all_placements = [
                self.generate_objects_per_room(args) for args in packed_args
            ]

        for placements in all_placements:
            results += placements

        return results

    def round_to_nearest_angle(self, angle, step=90):
        rounded_angle = round(angle / step) * step
        return rounded_angle % 360
    
    def generate_objects_per_room(self, args):
        plan_json, room, _, _, _, selected_objects, _ = args

        selected_floor_objects = selected_objects[room["roomType"]]["floor"]
        object_name2id = dict(selected_floor_objects)

        room_id = room["id"]
        room_origin = [
            min(v[0] for v in room["vertices"]),
            min(v[1] for v in room["vertices"]),
        ]

        with open(plan_json, 'r') as f:
            plan_objects = json.load(f)['object']

        placements = []
        for object_name, object_id in object_name2id.items():
            if object_name not in plan_objects:
                continue
                
            object_info = plan_objects[object_name]
            if object_info.get("location") != "floor":
                continue

            dim = get_bbox_dims(self.database[object_id])
            pos = object_info["position"]
            rot_y = self.round_to_nearest_angle(object_info["degree"])

            placements.append({
                **self.json_template,
                "id": f"{object_name} ({room_id})",
                "object_name": object_name,
                "assetId": object_id,
                "dimension": dim,
                "roomId": room_id,
                "position": {
                    "x": room_origin[0] + pos[0],
                    "y": dim["y"] / 2,
                    "z": room_origin[1] + pos[2],
                },
                "rotation": {"x": 0, "y": rot_y, "z": 0}
            })

        placements = nms(placements)
        
        align_scene_objects_to_walls(placements, room["vertices"])
        adjust_chairs_placement(placements, room["vertices"])
        optimize_object_positions(placements, room["vertices"])
        add_vertices_to_placement(placements)

        return placements

    def get_room_size(self, room):
        floor_polygon = room["floorPolygon"]
        x_values = [point["x"] for point in floor_polygon]
        z_values = [point["z"] for point in floor_polygon]
        return (
            int(max(x_values) - min(x_values)) * 100,
            int(max(z_values) - min(z_values)) * 100,
        )

    def solution2placement(self, solutions, object_name2id, room_id):
        placements = []
        for object_name, solution in solutions.items():
            if (
                "door" in object_name
                or "window" in object_name
                or "open" in object_name
            ):
                continue
            dimension = get_bbox_dims(self.database[object_name2id[object_name]])
            placement = self.json_template.copy()
            placement["assetId"] = object_name2id[object_name]
            placement["id"] = f"{object_name} ({room_id})"
            placement["position"] = {
                "x": solution[0][0] / 100,
                "y": dimension["y"] / 2,
                "z": solution[0][1] / 100,
            }
            placement["rotation"] = {"x": 0, "y": solution[1], "z": 0}
            placement["roomId"] = room_id
            placement["vertices"] = list(solution[2])
            placement["object_name"] = object_name
            placements.append(placement)
        return placements
