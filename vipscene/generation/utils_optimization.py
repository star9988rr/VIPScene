import math
from collections import defaultdict

from shapely.geometry import Polygon


def find_best_place_for_door(scene):
    wall_dict = {wall['id']: wall for wall in scene['walls']}
    excluded_directions = {
        wall_dict[wall_id]['direction']
        for window in scene['windows']
        for wall_id in [window['wall0'], window['wall1']]
        if wall_id in wall_dict
    }
    
    opposite_directions = {'north': 'south', 'south': 'north',
                          'east': 'west', 'west': 'east'}
    priority_directions = {opposite_directions[d] for d in excluded_directions
                          if d in opposite_directions}

    priority_walls = []
    other_walls = []
    for wall in scene['walls']:
        if "exterior" in wall['id']:
            continue
        if wall['direction'] in priority_directions:
            priority_walls.append(wall)
        elif wall['direction'] not in excluded_directions:
            other_walls.append(wall)

    best_wall, best_interval = process_wall_group(priority_walls, scene)
    if best_wall is None:
        best_wall, best_interval = process_wall_group(other_walls, scene)

    return (best_wall['id'], best_interval) if best_wall else (None, None)


def process_wall_group(walls, scene):
    best_wall = None
    best_interval = None
    max_length = 0
    prox_th = 0.4
    buffer_th = 0.01

    for wall in walls:
        p1, p2 = wall['segment']
        is_vertical = p1[0] == p2[0]
        coord_idx = 1 if is_vertical else 0
        wall_start = min(p1[coord_idx], p2[coord_idx])
        wall_end = max(p1[coord_idx], p2[coord_idx])
        wall_length = wall_end - wall_start
        wall_coord = p1[1 - coord_idx]
        direction = wall['direction']

        occupied_intervals = []
        for obj in scene['floor_objects']:
            pos, dim = obj['position'], obj['dimension']
            rotation_y = obj['rotation']['y'] % 360
            
            dx, dz = dim['x'], dim['z']
            eff_dx, eff_dz = (dz, dx) if rotation_y in [90, 270] else (dx, dz)
            
            x_min, x_max = pos['x'] - eff_dx/2, pos['x'] + eff_dx/2
            z_min, z_max = pos['z'] - eff_dz/2, pos['z'] + eff_dz/2

            if is_vertical:
                if (direction == 'east' and x_max < wall_coord - prox_th) or \
                   (direction == 'west' and x_min > wall_coord + prox_th):
                    continue
                proj_min, proj_max = z_min, z_max
            else:
                if (direction == 'north' and z_max < wall_coord - prox_th) or \
                   (direction == 'south' and z_min > wall_coord + prox_th):
                    continue
                proj_min, proj_max = x_min, x_max

            proj_start = max(wall_start, proj_min)
            proj_end = min(wall_end, proj_max)
            if proj_start < proj_end:
                buffer_start = max(wall_start, proj_start - buffer_th)
                buffer_end = min(wall_end, proj_end + buffer_th)
                if buffer_start < buffer_end:
                    occupied_intervals.append((buffer_start, buffer_end))

        occupied_intervals.sort()
        merged = []
        for start, end in occupied_intervals:
            if merged and merged[-1][1] >= start:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        available = []
        prev_end = wall_start
        for start, end in merged:
            if start > prev_end:
                available.append([prev_end, start])
            prev_end = max(prev_end, end)
        if prev_end < wall_end:
            available.append([prev_end, wall_end])

        for interval in available:
            length = interval[1] - interval[0]
            if length > max_length:
                max_length = length
                if direction in ['south', 'east']:
                    interval[0], interval[1] = wall_length - interval[1], wall_length - interval[0]
                best_interval = interval
                best_wall = wall

    return (best_wall, best_interval) if best_wall and max_length >= 1.1 else (None, None)


def distance_to_wall(x, y, A, B, C):
    return abs(A * x + B * y + C) / math.sqrt(A**2 + B**2)


def get_wall_equation(corner1, corner2):
    x1, y1 = corner1
    x2, y2 = corner2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def get_object_corners(position, dimension, rotation):
    x, z = position
    length, width = dimension["x"], dimension["z"]
    theta = math.radians(rotation)

    half_length = length / 2
    half_width = width / 2
    corners = [
        (x - half_length, z - half_width),
        (x + half_length, z - half_width),
        (x + half_length, z + half_width),
        (x - half_length, z + half_width),
    ]

    rotated_corners = []
    for corner in corners:
        dx = corner[0] - x
        dz = corner[1] - z
        new_x = x + dx * math.cos(theta) - dz * math.sin(theta)
        new_z = z + dx * math.sin(theta) + dz * math.cos(theta)
        rotated_corners.append((new_x, new_z))

    return rotated_corners


def get_bbox_on_xz_plane(position, dimension, rotation=0):
    """Get bounding box on XZ plane."""
    x, z = position
    dx, dz = dimension.get('x', 0), dimension.get('z', 0)
    
    # Handle 90 degree rotations
    rot_y = int(rotation) % 360
    if rot_y in [90, 270]:
        dx, dz = dz, dx

    half_dx, half_dz = dx / 2, dz / 2
    return (x - half_dx, z - half_dz, x + half_dx, z + half_dz)


def add_vertices_to_placement(placements):
    """Add vertices to placement objects, respecting rotation."""
    for placement in placements:
        pos = placement["position"]
        dim = placement["dimension"]
        position = (pos['x'] * 100, pos['z'] * 100)
        dimension = {k: v * 100 for k, v in dim.items()}
        rotation = placement["rotation"]["y"]
        
        vertices = get_object_corners(position, dimension, rotation)
        # Use Polygon to correctly represent rotated shape
        poly = Polygon(vertices)
        placement["vertices"] = list(poly.exterior.coords)


def compute_iou(box1, box2):
    """Compute Intersection over Union for two axis-aligned bounding boxes."""
    x_min1, z_min1, x_max1, z_max1 = box1
    x_min2, z_min2, x_max2, z_max2 = box2
    
    inter_x_min = max(x_min1, x_min2)
    inter_z_min = max(z_min1, z_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_z_max = min(z_max1, z_max2)
    
    if inter_x_max <= inter_x_min or inter_z_max <= inter_z_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_z_max - inter_z_min)
    area1 = (x_max1 - x_min1) * (z_max1 - z_min1)
    area2 = (x_max2 - x_min2) * (z_max2 - z_min2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_dist(box1, box2):
    """Compute distance between centers of two bounding boxes."""
    x_min1, z_min1, x_max1, z_max1 = box1
    x_min2, z_min2, x_max2, z_max2 = box2
    center1 = ((x_min1 + x_max1) / 2, (z_min1 + z_max1) / 2)
    center2 = ((x_min2 + x_max2) / 2, (z_min2 + z_max2) / 2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def nms_windows(placements, dist_threshold=1.0, iou_threshold=0.1):
    """Non-maximum suppression for window placements."""
    boxes = [get_bbox_on_xz_plane(
        (p["position"]["x"], p["position"]["z"]), 
        p["dimension"], p["rotation"]["y"]
    ) for p in placements]
    
    indices = sorted(range(len(boxes)), 
                     key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), 
                     reverse=True)
    
    keep = []
    for i in indices:
        current_box = boxes[i]
        current_name = placements[i]["object_name"].split("_")[-1]
        
        should_keep = True
        for kept_idx in keep:
            kept_box = boxes[kept_idx]
            if (compute_dist(current_box, kept_box) < dist_threshold or 
                compute_iou(current_box, kept_box) > iou_threshold):
                kept_name = placements[kept_idx]["object_name"].split("_")[-1]
                if kept_name == current_name:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(i)
    
    return [placements[i] for i in keep]


def adjust_windows_to_walls(placements, room_corners, th=1.0):
    """Adjust window placements to align with nearest walls."""
    walls = [
        (room_corners[0], room_corners[1]),
        (room_corners[1], room_corners[2]),
        (room_corners[2], room_corners[3]),
        (room_corners[3], room_corners[0]),
    ]
    room_size = room_corners[2]

    for placement in placements:
        pos = placement["position"]
        dim = placement["dimension"]
        rotation = placement["rotation"]["y"] % 360
        
        dist_wall = [
            distance_to_wall(pos["x"], pos["z"], *get_wall_equation(wall[0], wall[1]))
            for wall in walls
        ]
        
        if rotation < 20 or 160 < rotation < 200 or rotation > 340:
            nearest_wall = 1 if dist_wall[1] < dist_wall[3] else 3
        elif 70 < rotation < 110 or 250 < rotation < 290:
            nearest_wall = 0 if dist_wall[0] < dist_wall[2] else 2
        else:
            nearest_wall = dist_wall.index(min(dist_wall))
        
        new_rotation = nearest_wall * 90
        if new_rotation == 0:
            new_position = [min(room_size[0]-th, max(th, pos["x"])), 0]
            new_size = [dim["x"], dim["y"]]
            direction = 'south'
        elif new_rotation == 90:
            new_position = [0, min(room_size[1]-th, max(th, pos["z"]))]
            new_size = [dim["z"], dim["y"]]
            direction = 'west'
        elif new_rotation == 180:
            new_position = [min(room_size[0]-th, max(th, pos["x"])), room_size[1]]
            new_size = [dim["x"], dim["y"]]
            direction = 'north'
        else:
            new_position = [room_size[0], min(room_size[1]-th, max(th, pos["z"]))]
            new_size = [dim["z"], dim["y"]]
            direction = 'east'

        placement["position"]["x"] = new_position[0]
        placement["position"]["z"] = new_position[1]
        placement["rotation"]["y"] = new_rotation
        placement["new_size"] = new_size
        placement["direction"] = direction


def nms(placements, iou_threshold_same_cls=0.1, iou_threshold=0.2):
    """Non-maximum suppression for object placements."""
    boxes = [
        get_bbox_on_xz_plane(
            (p["position"]["x"], p["position"]["z"]),
            p["dimension"],
            p["rotation"]["y"]
        )
        for p in placements
    ]

    def bbox_area(bbox):
        x_min, z_min, x_max, z_max = bbox
        return (x_max - x_min) * (z_max - z_min)

    sorted_indices = sorted(range(len(boxes)), key=lambda i: bbox_area(boxes[i]), reverse=True)
    placements = [placements[i] for i in sorted_indices]
    boxes = [boxes[i] for i in sorted_indices]
    
    final_placements = []
    keep = []
    for i in range(len(placements)):
        current_box = boxes[i]
        current_name = placements[i]["object_name"].split("_")[-1]
        
        should_keep = True
        for kept_idx in keep:
            iou = compute_iou(current_box, boxes[kept_idx])
            
            kept_name = placements[kept_idx]["object_name"].split("_")[-1]
            is_name_match = (kept_name == current_name or 
                             kept_name in current_name or 
                             current_name in kept_name)
            
            threshold = iou_threshold_same_cls if is_name_match else iou_threshold
            
            if iou > threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(i)
            final_placements.append(placements[i])

    return final_placements


def snap_object_to_wall(position, dimension, room_size, nearest_wall, offset=0.1):
    if nearest_wall == 0:
        new_x = dimension["z"] / 2 + offset
        new_z = position[1]
        new_rotation = 90
    elif nearest_wall == 1:
        new_x = position[0]
        new_z = room_size[1] - dimension["z"] / 2 - offset
        new_rotation = 180
    elif nearest_wall == 2:
        new_x = room_size[0] - dimension["z"] / 2 - offset
        new_z = position[1]
        new_rotation = 270
    elif nearest_wall == 3:
        new_x = position[0]
        new_z = dimension["z"] / 2 + offset
        new_rotation = 0

    return [new_x, new_z], new_rotation


def align_scene_objects_to_walls(placements, room_corners, threshold=2.0):
    """
    Adjust objects near walls to align edges with walls and make them parallel.
    """
    # Define the four walls of the room
    walls = [
        (room_corners[0], room_corners[1]),
        (room_corners[1], room_corners[2]),
        (room_corners[2], room_corners[3]),
        (room_corners[3], room_corners[0]),
    ]

    room_size = room_corners[2]
    # Iterate through each object
    for placement in placements:
        object_name = placement["object_name"]
        if "countertop" in object_name or 'island' in object_name or 'table' in object_name:
            continue
        position = placement["position"]
        dimension = placement["dimension"]  # Assume dimension contains length and width
        rotation = placement["rotation"]["y"]  # Assume rotation angle is on the y-axis

        if "bed" in object_name:
            if rotation in [0, 180]:
                if distance_to_wall(position["x"], position["z"], *get_wall_equation(*walls[1])) < distance_to_wall(position["x"], position["z"], *get_wall_equation(*walls[3])):
                    nearest_wall = 1
                else:
                    nearest_wall = 3

            elif rotation in [90, 270]:
                if distance_to_wall(position["x"], position["z"], *get_wall_equation(*walls[0])) < distance_to_wall(position["x"], position["z"], *get_wall_equation(*walls[2])):
                    nearest_wall = 0
                else:
                    nearest_wall = 2

            new_position, new_rotation = snap_object_to_wall(
                (position["x"], position["z"]), dimension, room_size, nearest_wall
            )
            placement["position"]["x"] = new_position[0]
            placement["position"]["z"] = new_position[1]
            placement["rotation"]["y"] = new_rotation

        else:
            min_distance_wall = float('inf')
            nearest_wall = None
            # Iterate through each wall
            for i in range(len(walls)):
                wall = walls[i]
                corner1, corner2 = wall
                A, B, C = get_wall_equation(corner1, corner2)

                dist = distance_to_wall(position["x"], position["z"], A, B, C)
                if dist < min_distance_wall:
                    min_distance_wall = dist
                    nearest_wall = i

            # If the object is near a wall
            if min_distance_wall < threshold:
                # Adjust object position and orientation
                new_position, new_rotation = snap_object_to_wall(
                    (position["x"], position["z"]), dimension, room_size, nearest_wall
                )
                placement["position"]["x"] = new_position[0]
                placement["position"]["z"] = new_position[1]
                placement["rotation"]["y"] = new_rotation


def adjust_chairs_placement(placements, room_corners):
    """Adjust chair placements around tables."""
    tables = []
    chairs = []
    for p in placements:
        name = p['object_name'].lower()
        if any(keyword in name for keyword in ['table', 'desk', 'countertop', 'island']):
            tables.append(p)
        elif ('chair' in name and 'armchair' not in name) or 'stool' in name:
            chairs.append(p)

    if not tables:
        if chairs:
            align_scene_objects_to_walls(chairs, room_corners)
        return

    for table in tables:
        t_center = table['position']
        t_dim = table['dimension']
        t_corners = get_object_corners((t_center['x'], t_center['z']), t_dim, table['rotation']['y'])
        t_size_x, t_size_z = t_dim['x'], t_dim['z']

        t_x_min = min(c[0] for c in t_corners)
        t_x_max = max(c[0] for c in t_corners)
        t_z_min = min(c[1] for c in t_corners)
        t_z_max = max(c[1] for c in t_corners)
        threshold = math.hypot(t_size_x, t_size_z) / 2 + 1.0

        chair_groups = defaultdict(list)
        for chair in chairs:
            c_pos = chair['position']
            dx = c_pos['x'] - t_center['x']
            dz = c_pos['z'] - t_center['z']
            
            if math.hypot(dx, dz) > threshold:
                align_scene_objects_to_walls([chair], room_corners)
                continue
            
            side = 'east' if dx > 0 else 'west' if abs(dx) > abs(dz) else 'north' if dz > 0 else 'south'
            chair_groups[side].append(chair)

        for side, group in chair_groups.items():
            if not group:
                continue

            chair_gap = 0.0
            chair_dim = group[0]['dimension']
            chair_depth = chair_dim['z']

            if side == 'north':
                base_z = t_z_max + chair_gap + chair_depth/2
                align_axis = 'x'
                align_range = (t_x_min, t_x_max)
            elif side == 'south':
                base_z = t_z_min - chair_gap - chair_depth/2
                align_axis = 'x'
                align_range = (t_x_min, t_x_max)
            elif side == 'east':
                base_x = t_x_max + chair_gap + chair_depth/2
                align_axis = 'z'
                align_range = (t_z_min, t_z_max)
            else:  # west
                base_x = t_x_min - chair_gap - chair_depth/2
                align_axis = 'z'
                align_range = (t_z_min, t_z_max)

            target_count = len(group)
            sorted_chairs = sorted(group, key=lambda c: abs(c['position'][align_axis] - sum(align_range)/2))
            selected_chairs = sorted_chairs[:target_count]

            step = (align_range[1] - align_range[0] - chair_dim['x']) / (target_count + 1)
            positions = [align_range[0] + (i+1)*step + chair_dim['x']/2 for i in range(target_count)]

            for idx, chair in enumerate(selected_chairs):
                if align_axis == 'x':
                    chair['position']['x'] = positions[idx]
                    chair['position']['z'] = base_z
                    chair['rotation']['y'] = 180 if side == 'north' else 0
                else:
                    chair['position']['z'] = positions[idx]
                    chair['position']['x'] = base_x
                    chair['rotation']['y'] = 270 if side == 'east' else 90

    return placements


def optimize_object_positions(objects, room_vertices, max_iterations=1000, learning_rate=0.01, position_weight=1.0, overlap_weight=10.0, boundary_weight=10.0):
    """
    Optimize positions of objects in the room to:
    1. Stay within room boundaries
    2. Avoid overlaps between objects
    3. Stay close to original positions
    """
    
    room_poly = Polygon(room_vertices)
    original_positions = {
        obj['id']: (obj['position']['x'], obj['position']['z']) 
        for obj in objects
    }

    def get_object_bbox_with_padding(obj):
        """Get object bounding box corners in XZ plane with 0.05m padding for overlap checking"""
        x, z = obj['position']['x'], obj['position']['z']
        rot_y = math.radians(obj['rotation']['y'])
        # Add 0.05m padding to each dimension (total gap of 0.1m between objects)
        dx = obj['dimension']['x']/2 + 0.05
        dz = obj['dimension']['z']/2 + 0.05
        
        corners = []
        for corner in [(dx,dz), (dx,-dz), (-dx,-dz), (-dx,dz)]:
            rx = corner[0]*math.cos(rot_y) - corner[1]*math.sin(rot_y)
            rz = corner[0]*math.sin(rot_y) + corner[1]*math.cos(rot_y)
            corners.append((x + rx, z + rz))
        return corners
    
    def get_object_bbox(obj):
        """Get object bounding box corners in XZ plane without padding for boundary checking"""
        x, z = obj['position']['x'], obj['position']['z']
        rot_y = math.radians(obj['rotation']['y'])
        dx = obj['dimension']['x']/2
        dz = obj['dimension']['z']/2
        
        corners = []
        for corner in [(dx,dz), (dx,-dz), (-dx,-dz), (-dx,dz)]:
            rx = corner[0]*math.cos(rot_y) - corner[1]*math.sin(rot_y)
            rz = corner[0]*math.sin(rot_y) + corner[1]*math.cos(rot_y)
            corners.append((x + rx, z + rz))
        return corners
        

    def compute_losses():
        """Compute position, overlap and boundary losses"""
        position_loss = 0
        overlap_loss = 0
        boundary_loss = 0
        
        # Position loss - distance from original position
        for obj in objects:
            orig_x, orig_z = original_positions[obj['id']]
            curr_x, curr_z = obj['position']['x'], obj['position']['z']
            position_loss += ((curr_x - orig_x)**2 + (curr_z - orig_z)**2)
        
        # Overlap loss between objects (using padded bounding boxes)
        for i, obj1 in enumerate(objects):
            bbox1 = Polygon(get_object_bbox_with_padding(obj1))
            # Check overlap with other objects
            for obj2 in objects[i+1:]:
                bbox2 = Polygon(get_object_bbox_with_padding(obj2))
                if bbox1.intersects(bbox2):
                    overlap_area = bbox1.intersection(bbox2).area

                    overlap_loss += overlap_area
                    
            # Boundary loss - objects outside room (using non-padded bounding boxes)
            bbox1_no_padding = Polygon(get_object_bbox(obj1))
            if not room_poly.contains(bbox1_no_padding):
                boundary_loss += bbox1_no_padding.difference(room_poly).area
        
                
        return position_loss, overlap_loss, boundary_loss

    # Optimization loop
    best_loss = float('inf')
    best_positions = None
    
    for iteration in range(max_iterations):
        # Compute current losses
        pos_loss, ovr_loss, bnd_loss = compute_losses()
        total_loss = (position_weight * pos_loss + 
                     overlap_weight * ovr_loss + 
                     boundary_weight * bnd_loss)
        
        # Save best configuration
        if total_loss < best_loss:
            best_loss = total_loss
            best_positions = {
                obj['id']: (obj['position']['x'], obj['position']['z'])
                for obj in objects
            }
            
        # Early stopping if losses are zero
        if ovr_loss == 0 and bnd_loss == 0:
            break
            
        # Update positions using pattern search
        for obj in objects:
            deltas = [(learning_rate, 0), (-learning_rate, 0), 
                        (0, learning_rate), (0, -learning_rate)]

            best_delta = None
            best_delta_loss = total_loss
            
            for dx, dz in deltas:
                # Try moving object
                orig_x, orig_z = obj['position']['x'], obj['position']['z']
                obj['position']['x'] += dx
                obj['position']['z'] += dz
                
                # Compute new loss
                new_pos_loss, new_ovr_loss, new_bnd_loss = compute_losses()
                new_total_loss = (position_weight * new_pos_loss +
                                overlap_weight * new_ovr_loss +
                                boundary_weight * new_bnd_loss)
                
                # Restore position
                obj['position']['x'] = orig_x
                obj['position']['z'] = orig_z
                
                # Update best move if loss improved
                if new_total_loss < best_delta_loss:
                    best_delta = (dx, dz)
                    best_delta_loss = new_total_loss
            
            # Apply best move
            if best_delta is not None:
                obj['position']['x'] += best_delta[0]
                obj['position']['z'] += best_delta[1]
    
    # Restore best positions found
    if best_positions is not None:
        for obj in objects:
            obj['position']['x'], obj['position']['z'] = best_positions[obj['id']]
            
    return objects
