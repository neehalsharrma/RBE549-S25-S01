import bpy
import json
import os
from typing import List, Tuple

# Dictionary to store relative filepaths for different vehicle types
# These filepaths point to .blend files containing 3D models of vehicles
vehicle_filepaths = {
    "Car": "./Assets/Vehicles/Car.blend",
    "Truck": "./Assets/Vehicles/Truck.blend",
    "SUV": "./Assets/Vehicles/SUV.blend",
    "Motorcycle": "./Assets/Vehicles/Motorcycle.blend",
    "Bus": "./Assets/Vehicles/Bus.blend",
    "Bicycle": "./Assets/Vehicles/Bicycle.blend",
    "Pickup Truck": "./Assets/Vehicles/PickupTruck.blend",
}

# Dictionary to store relative filepaths for different traffic items
# These filepaths point to .blend files containing 3D models of traffic-related objects
traffic_item_filepaths = {
    "Traffic Signal": "./Assets/TrafficItems/TrafficSignal.blend",
    "Dustbin": "./Assets/TrafficItems/Dustbin.blend",
    "Traffic Cone": "./Assets/TrafficItems/TrafficCone.blend",
    "Speed Limit": "./Assets/TrafficItems/SpeedLimitSign.blend",
}


def spawn_objects(
    filepath: str,
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
) -> List[bpy.types.Object]:
    """
    Spawns objects from a specified .blend file into the current Blender scene.
    This function loads objects from a given .blend file, creates copies of the
    top-level objects and their children, and places them in the current scene
    at the specified location and rotation. The objects are scaled uniformly
    by a factor of 0.02.

    Parameters
    ----------
    filepath : str
        The file path to the .blend file containing the objects to be spawned.
    location : tuple of float
        A tuple (x, y, z) specifying the location where the objects will be placed.
    rotation : tuple of float
        A tuple (rx, ry, rz) specifying the rotation (in radians) to be applied
        to the objects.

    Returns
    -------
    list of bpy.types.Object
        A list of the spawned objects, including both top-level objects and their children.
    """
    # Load objects from the specified .blend file
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects

    spawned_objects = []  # List to store the spawned objects

    # Iterate through the objects in the .blend file
    for obj in data_to.objects:
        if obj.parent is None:  # Only process top-level objects
            # Create a copy of the object and its data
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            new_obj.animation_data_clear()  # Clear any animation data
            new_obj.location = location  # Set the object's location
            new_obj.rotation_euler = rotation  # Set the object's rotation
            new_obj.scale = (0.02, 0.02, 0.02)  # Scale the object
            bpy.context.collection.objects.link(new_obj)  # Link the object to the scene
            spawned_objects.append(new_obj)

            # Process and link child objects
            for child_obj in obj.children:
                new_child_obj = child_obj.copy()
                new_child_obj.data = child_obj.data.copy()
                new_child_obj.animation_data_clear()
                new_child_obj.parent = new_obj  # Maintain parent-child relationship
                bpy.context.collection.objects.link(new_child_obj)
                spawned_objects.append(new_child_obj)

    return spawned_objects


# Delete all objects in the scene except for the camera
# This ensures a clean scene before adding new objects
for obj in bpy.context.scene.objects:
    if obj.type != "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

# Path to the JSON file containing object spawn data
file_path = "./spawn.json"

# Load the JSON file containing object details
with open(file_path, "r") as file:
    data = json.load(file)

# Initialize lists to store object details for each frame
frames: List[int] = []  # Frame numbers
types: List[str] = []  # Object types
positions: List[dict] = []  # Object positions
rotations: List[dict] = []  # Object rotations
scales: List[dict] = []  # Object scales

# Spawn a car object at the origin with a specific rotation
spawn_objects(vehicle_filepaths["Car"], (0, 0, 0), (0, 0, 3.14))

# Set the camera's location and rotation
camera_location = (0.0, 0.2, 1.4)  # Camera position
camera_rotation = (1.57, 0.0, 0.0)  # Camera rotation in radians

# Check if a camera exists in the scene and set its properties
if "Camera" in bpy.data.objects:
    camera = bpy.data.objects["Camera"]
    camera.location = camera_location
    camera.rotation_euler = camera_rotation

# Iterate through the JSON data to extract object details
for frame_data in data:
    for obj in frame_data["objects"]:
        types.append(obj["type"])  # Store object type
        positions.append(obj["position"])  # Store object position
        rotations.append(obj["rotation"])  # Store object rotation
        scales.append(obj["scale"])  # Store object scale

    # Spawn objects based on the extracted positions and rotations
    for i, position in enumerate(positions):
        x, y, z = positions[i]["x"], positions[i]["y"], 0  # Extract position
        phi, theta, psi = (
            rotations[i]["x"],
            rotations[i]["y"],
            rotations[i]["z"],
        )  # Extract rotation

        # Check if the object type is recognized and get the corresponding .blend file
        if types[i] in vehicle_filepaths:
            blend_filepath = vehicle_filepaths[types[i]]
        elif types[i] in traffic_item_filepaths:
            blend_filepath = traffic_item_filepaths[types[i]]
        else:
            continue  # Skip if the type is not recognized
        spawn_objects(blend_filepath, (x, y, z), (phi, theta, psi))

    # Render the scene and save the image
    bpy.ops.render.render(write_still=True)  # Render the scene
    image_name = f"{frame_data['frame']:06d}.png"  # Generate the image name using the frame number
    image_filepath = os.path.join("./Results/", image_name)  # Construct the file path
    bpy.data.images["Render Result"].save_render(
        image_filepath
    )  # Save the rendered image
    print(f"Rendered image {image_name}")  # Print confirmation
