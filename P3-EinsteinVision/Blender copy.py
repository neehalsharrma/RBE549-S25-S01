"""
This script automates the process of spawning 3D objects in a Blender scene based on data
provided in a JSON file. It supports various types of vehicles and traffic-related items,
loading their models from specified .blend files. The script also handles object placement,
rotation, scaling, and rendering of the scene.

Key Features:
- Load 3D models from .blend files for different object types.
- Spawn objects at specified locations with given rotations and scales.
- Render the scene and save the output images for each frame.
- Clean the scene by removing all objects except the camera before spawning new objects.

Dependencies:
- Blender's Python API (bpy)
- JSON file containing object spawn data
"""

import json
import os
import sys
from typing import Tuple, List

import bpy
from pyffmpeg import FFmpeg  # Import pyffmpeg

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True

# Dictionary to store absolute filepaths for different vehicle types
# These filepaths point to .blend files containing 3D models of vehicles
vehicle_filepaths = {
    "car": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/Car.blend",
    "truck": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/Truck.blend",
    "suv": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/SUV.blend",
    "motorcycle": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/Motorcycle.blend",
    "bus": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/Bus.blend",
    "bicycle": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/Bicycle.blend",
    "pickup truck": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/PickupTruck.blend",
}

# Dictionary to store absolute filepaths for different traffic items
# These filepaths point to .blend files containing 3D models of traffic-related objects
traffic_item_filepaths = {
    "traffic light": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/TrafficLight.blend",
    "dustbin": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Dustbin.blend",
    "traffic cone": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/TrafficCone.blend",
    "speed limit": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/SpeedLimitSign.blend",
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
            if filepath in vehicle_filepaths.values():
                new_obj.rotation_euler = rotation  # Set the object's rotation
            else:
                new_obj.rotation_euler = (
                    rotation[0] + 1.57,
                    rotation[1],
                    rotation[2] - 1.57,
                )  # Rotate x by 90 degrees
            if filepath in vehicle_filepaths.values():
                if "Truck.blend" in filepath:  # Check if the vehicle is a truck
                    new_obj.scale = (0.000964, 0.000964, 0.000964)  # Scale for truck
                else:
                    new_obj.scale = (0.02, 0.02, 0.02)  # Scale for other vehicles
            else:
                new_obj.scale = (
                    0.5,
                    0.5,
                    0.5,
                )  # Scale the object to 0.5 if it's not a vehicle
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


def render_scene(data: List[dict], render_output_dir: str, video_number: int) -> None:
    """
    Render the Blender scene based on the provided data and save the output images.

    Parameters
    ----------
    data : list of dict
        List of frame data containing object details.
    render_output_dir : str
        Directory to save the rendered images.
    video_number : int
        Video number for organizing output directories.

    Returns
    -------
    None
    """
    for frame_data in data:
        # Delete all objects in the scene except for the camera
        for obj in bpy.context.scene.objects:
            if obj.type != "CAMERA":
                bpy.data.objects.remove(obj, do_unlink=True)

        # Spawn a car object at the origin with a specific rotation
        spawn_objects(vehicle_filepaths["car"], (0, 0, 0), (0, 0, 3.14))

        # Set the camera's location and rotation
        camera_location = (0.0, 0.2, 1.4)
        camera_rotation = (1.57, 0.0, 0.0)
        if "Camera" in bpy.data.objects:
            camera = bpy.data.objects["Camera"]
            camera.location = camera_location
            camera.rotation_euler = camera_rotation

        # Extract object details for the current frame
        for obj in frame_data["objects"]:
            x, y, z = obj["position"]["x"], obj["position"]["y"], 0
            phi, theta, psi = (
                obj["rotation"]["x"],
                obj["rotation"]["y"],
                obj["rotation"]["z"],
            )

            if obj["type"] in vehicle_filepaths:
                blend_filepath = vehicle_filepaths[obj["type"]]
            elif obj["type"] in traffic_item_filepaths:
                blend_filepath = traffic_item_filepaths[obj["type"]]
            else:
                continue
            spawn_objects(blend_filepath, (x, y, z), (phi, theta, psi))

        # Render the scene and save the image
        bpy.ops.render.render(write_still=True)
        image_name = f"{frame_data['frame']:06d}.png"
        image_filepath = os.path.join(render_output_dir, image_name)
        bpy.data.images["Render Result"].save_render(image_filepath)
        print(f"Rendered image {image_name}")

    # Generate a video from the rendered images using pyffmpeg
    output_video_path = f"/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Results/FFMPEG/vid_{video_number}/blender_render_video.mp4"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    ff = FFmpeg()
    input_pattern = os.path.join(render_output_dir, "%06d.png")
    ff.options(
        f"-y -framerate 30 -i {input_pattern} -c:v libx264 -pix_fmt yuv420p {output_video_path}"
    )
    print(f"Blender render video saved at {output_video_path}")


def main() -> None:
    """
    Main function to load data, render the scene, and generate the video.

    Returns
    -------
    None
    """
    # Path to the JSON file containing object spawn data
    file_path = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/spawn.json"

    # Load the JSON file containing object details
    with open(file_path, "r") as file:
        data = json.load(file)

    # Get the video number from the Wrapper.py script
    video_number = 5  # Ensure this matches the value in Wrapper.py

    # Define the output directory for renders based on the video number
    render_output_dir = f"/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Results/Renders/vid_{video_number}"
    os.makedirs(render_output_dir, exist_ok=True)

    # Render the scene and generate the video
    render_scene(data, render_output_dir, video_number)


if __name__ == "__main__":
    main()
