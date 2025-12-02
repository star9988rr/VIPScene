floor_plan_prompt = """You are an experienced room designer. Please assist me in crafting a floor plan. Each room is a rectangle. You need to specify an appropriate design scheme, including each room's color, material, and texture.
For example:
living room | maple hardwood, matte | light grey drywall, smooth
kitchen | white hex tile, glossy | light grey drywall, smooth

Here are some guidelines for you:
1. It is okay to have one room in the floor plan if you think it is reasonable.
2. The room name should be unique.

Now, I need a design for {input}.
Additional requirements: {additional_requirements}.
Your response should be direct and without additional text at the beginning or end."""


wall_height_prompt = """I am now designing {input}. Please help me decide the wall height in meters.
Answer with a number, for example, 3.0. Do not add additional text at the beginning or in the end."""


doorway_prompt = """I need assistance in designing the connections between rooms. The connections could be of three types: doorframe (no door installed), doorway (with a door), or open (no wall separating rooms). The sizes available for doorframes and doorways are single (1m wide) and double (2m wide).

Ensure that the door style complements the design of the room. The output format should be: room 1 | room 2 | connection type | size | door style. For example:
exterior | living room | doorway | double | dark brown metal door
living room | kitchen | open | N/A | N/A
living room | bedroom | doorway | single | wooden door with white frames

The design under consideration is {input}, which includes these rooms: {rooms}. The length, width and height of each room in meters are:
{room_sizes}
Certain pairs of rooms share a wall: {room_pairs}. There must be a door to the exterior.
Adhere to these additional requirements: {additional_requirements}.
Provide your response succinctly, without additional text at the beginning or end."""


window_prompt = """Guide me in designing the windows for each room. The window types are: fixed, hung, and slider.
The available sizes (width x height in cm) are:
fixed: (92, 120), (150, 92), (150, 120), (150, 180), (240, 120), (240, 180)
hung: (87, 160), (96, 91), (120, 160), (130, 67), (130, 87), (130, 130)
slider: (91, 92), (120, 61), (120, 91), (120, 120), (150, 92), (150, 120)

Your task is to determine the appropriate type, size, and quantity of windows for each room, bearing in mind the room's design, dimensions, and function.

Please format your suggestions as follows: room | wall direction | window type | size | quantity | window base height (cm from floor). For example:
living room | west | fixed | (130, 130) | 1 | 50

I am now designing {input}. The wall height is {wall_height} cm. The walls available for window installation (direction, width in cm) in each room are:
{walls}
Please note: It is not mandatory to install windows on every available wall. Within the same room, all windows must be the same type and size.
Also, adhere to these additional requirements: {additional_requirements}.

Provide a concise response, omitting any additional text at the beginning or end. """




wall_object_constraints_prompt = """You are an experienced room designer.
Please help me arrange wall objects in the room by providing their relative position and distance from the floor.
The output format must be: wall object | above, floor object  | distance from floor (cm). For example:
painting | above, sofa | 160
switch | N/A | 120
Note the distance is the distance from the *bottom* of the wall object to the floor. The second column is optional and can be N/A. The object of the same type should be placed at the same height.
Now I am designing {room_type} of which the wall height is {wall_height} cm, and the floor objects in the room are: {floor_objects}.
The wall objects I want to place in the {room_type} are: {wall_objects}.
Please do not add additional text at the beginning or in the end."""


ceiling_selection_prompt = """Assist me in selecting ceiling objects (light/fan) to furnish each room.
Present your recommendations in this format: room type | ceiling object description
For example:
living room | modern, 3-light, semi-flush mount ceiling light

Currently, the design in progress is "{input}", featuring these rooms: {rooms}. You need to provide one ceiling object for each room.
Please also consider the following additional requirements: {additional_requirements}.

Your response should be precise, without additional text at the beginning or end. """


wall_object_selection_prompt_json = """You are an experienced room designer, please assist me in selecting wall-based objects to furnish the room. You need to select appropriate objects to satisfy the customer's requirements.
You must provide a description and desired size for each object since I will use it to retrieve object.
Present your recommendations in JSON format:
{
    object_name:{
        "description": a short sentence describing the object,
        "location": "wall",
        "size": the desired size of the object, in the format of a list of three numbers, [length, width, height] in centimeters,
        "quantity": 1,
        "variance_type": "same",
        "objects_on_top": a list of small children objects (can be empty) which are placed *on top of* this object. For each child object, you only need to provide the object name, quantity and variance type. For example, {"object_name": "book", "quantity": 2, "variance_type": "varied"}
    }
}

For example:
{
    "painting": {
        "description": "abstract painting",
        "location": "wall",
        "size": [100, 100, 5],
        "quantity": 1,
        "variance_type": "same",
        "objects_on_top": []
    },
    "wall shelf": {
        "description": "a modern style wall shelf",
        "location": "wall",
        "size": [100, 30, 50],
        "quantity": 1,
        "variance_type": "same",
        "objects_on_top": []
    }
}

We are working on the room with the size of ROOM_SIZE.
Please also consider the following requirements: REQUIREMENTS.

Here are some guidelines for you:
1. Provide two or three reasonable objects for each room.
2. Do not provide rug/mat, windows, doors, curtains, and ceiling objects which have been installed for each room.
3. Do not provide shelves/desks on the wall.

Please first use natural language to explain your high-level design strategy, and then follow the desired JSON format *strictly* (do not add any additional text at the beginning or end)."""





small_object_selection_prompt_json = """As an experienced room designer, you are tasked to bring life into the room by strategically placing more *small* objects. Those objects should only be arranged *on top of* large objects which serve as receptacles. 
Present your recommendations in JSON format:
{
    object_name:{
        "objects_on_top": a list of small children objects (can be empty) which are placed *on top of* this object. For each child object, you only need to provide the object name, quantity and variance type. For example, {"object_name": "book", "quantity": 2, "variance_type": "varied"}
    }
}
Here, the variance type specifies whether the small objects are same or varied. There's no restriction on the number of small objects you can select for each receptacle. An example of this format is as follows:

For example:
{
    "sofa": {
        "objects_on_top": [
            {"object_name": "news paper", "quantity": 2, "variance_type": "varied"},
            {"object_name": "pillow", "quantity": 2, "variance_type": "varied"},
            {"object_name": "mobile phone", "quantity": 1, "variance_type": "same"}
        ]
    },
    "tv stand": {
        "objects_on_top": [
            {"object_name": "49 inch TV", "quantity": 1, "variance_type": "same"},
            {"object_name": "speaker", "quantity": 2, "variance_type": "same"},
            {"object_name": "remote control for TV", "quantity": 1, "variance_type": "same"}
        ]
    }
}

Now, we are designing ADDITIONAL_REQUIREMENTS.
The available receptacles in the room include: RECEPTACLES.
Your response should solely contain the information about the placement of objects and should not include any additional text before or after the main content."""

