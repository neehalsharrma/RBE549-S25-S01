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
import argparse
from math import degrees
from typing import List, Tuple
import glob

import bpy
from pyffmpeg import FFmpeg

# Disable the creation of __pycache__ directories
sys.dont_write_bytecode = True

# Dictionary to store absolute filepaths for different vehicle types
# These filepaths point to .blend files containing 3D models of vehicles
vehicle_filepaths = {
    "car": "./Assets/Vehicles/Car.blend",
    "van": "./Assets/Vehicles/Truck.blend",
    "suv": "./Assets/Vehicles/SUV.blend",
    "motorcycle": "./Assets/Vehicles/Motorcycle.blend",
    "bus": "./Assets/Vehicles/Bus.blend",
    "bicycle": "./Assets/Vehicles/Bicycle.blend",
    "pickup truck": "./Assets/Vehicles/PickupTruck.blend",
}

# Dictionary to store absolute filepaths for different traffic items
# These filepaths point to .blend files containing 3D models of traffic-related objects
traffic_item_filepaths = {
    "traffic light": "./Assets/TrafficLight.blend",
    "dustbin": "./Assets/Dustbin.blend",
    "traffic cone": "./Assets/TrafficCone.blend",
    "speed limit": "./Assets/SpeedLimitSign.blend",
    "stop sign": "./Assets/StopSign.blend",
}

# Global constants
SPAWN_JSON_PATH = "./spawn.json"
WRAPPER_SCRIPT_PATH = "./Wrapper.py"
RENDER_OUTPUT_BASE_DIR = "./Results/Renders"
FFMPEG_OUTPUT_BASE_DIR = "./Results/FFMPEG"
VIDEO_NUMBER = 1  # Set the video number for output organization
STOP_SIGN_TEXTURE_PATH = "./Assets/StopSignImage.png"  # Path to stop sign texture
SPEED_LIMIT_TEXTURE_PATH = (
    "./Assets/SpeedLimitSignImage.png"  # Path to speed limit texture
)


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
        data_to.objects = [
            obj for obj in data_from.objects if obj not in {"Camera", "Sun"}
        ]

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
            bpy.context.collection.objects.link(new_obj)  # Link the object to the scene
            spawned_objects.append(new_obj)

            # Process and link child objects
            for child_obj in obj.children:
                new_child_obj = child_obj.copy()
                new_child_obj.data = child_obj.data.copy()
                new_child_obj.animation_data_clear()
                new_child_obj.parent = new_obj
                bpy.context.collection.objects.link(new_child_obj)
                spawned_objects.append(new_child_obj)

    return spawned_objects


def setup_compositing_nodes():
    """
    Sets up compositing nodes in Blender to overlay stop sign and speed limit images
    on the rendered frames.

    Returns
    -------
    None
    """
    # Enable compositing and clear existing nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create render layers node
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    render_layers.location = (0, 0)

    # Create image nodes
    stop_sign = tree.nodes.new(type="CompositorNodeImage")
    stop_sign.location = (0, 200)
    stop_sign.name = "StopSignImage"
    # To load your image:
    stop_sign.image = bpy.data.images.load(STOP_SIGN_TEXTURE_PATH)

    speed_limit = tree.nodes.new(type="CompositorNodeImage")
    speed_limit.location = (0, -200)
    speed_limit.name = "SpeedLimitSign"
    # To load your image:
    speed_limit.image = bpy.data.images.load(SPEED_LIMIT_TEXTURE_PATH)

    # Create transform nodes
    transform1 = tree.nodes.new(type="CompositorNodeTransform")
    transform1.location = (300, 200)
    transform1.inputs["X"].default_value = 0.0
    transform1.inputs["Y"].default_value = 350.0
    transform1.inputs["Angle"].default_value = 0.0
    transform1.inputs["Scale"].default_value = 0.200
    transform1.filter_type = "NEAREST"

    transform2 = tree.nodes.new(type="CompositorNodeTransform")
    transform2.location = (300, -200)
    transform2.inputs["X"].default_value = 270.0
    transform2.inputs["Y"].default_value = 350.0
    transform2.inputs["Angle"].default_value = 0.0
    transform2.inputs["Scale"].default_value = 0.090
    transform2.filter_type = "NEAREST"

    # Create Alpha Over nodes
    alpha_over1 = tree.nodes.new(type="CompositorNodeAlphaOver")
    alpha_over1.name = "AlphaOver_StopSign"
    alpha_over1.location = (600, 100)
    alpha_over1.use_premultiply = True
    alpha_over1.premul = 0.0
    alpha_over1.inputs[0].default_value = 0.1  # Fac value

    alpha_over2 = tree.nodes.new(type="CompositorNodeAlphaOver")
    alpha_over2.name = "AlphaOver_SpeedLimit"
    alpha_over2.location = (900, 0)
    alpha_over2.use_premultiply = True
    alpha_over2.premul = 0.0
    alpha_over2.inputs[0].default_value = 0.1  # Fac value

    # Create Composite output node
    composite = tree.nodes.new(type="CompositorNodeComposite")
    composite.location = (1200, 0)
    composite.use_alpha = True

    # Connect nodes
    links = tree.links

    # Connect stop sign image to transform1
    links.new(stop_sign.outputs["Image"], transform1.inputs[0])

    # Connect speed limit image to transform2
    links.new(speed_limit.outputs["Image"], transform2.inputs[0])

    # Connect render layers to first alpha over
    links.new(render_layers.outputs["Image"], alpha_over1.inputs[1])

    # Connect transform1 output to first alpha over
    links.new(transform1.outputs[0], alpha_over1.inputs[2])

    # Connect first alpha over to second alpha over
    links.new(alpha_over1.outputs[0], alpha_over2.inputs[1])

    # Connect transform2 output to second alpha over
    links.new(transform2.outputs[0], alpha_over2.inputs[2])

    # Connect final result to composite output
    links.new(alpha_over2.outputs[0], composite.inputs[0])


def add_pedestrians_to_scene(frame_number: int):
    """
    Search for pedestrian .obj files corresponding to the given frame number
    and add them to the Blender scene with specified transformations.

    Parameters
    ----------
    frame_number : int
        The frame number to search for corresponding pedestrian files.

    Returns
    -------
    None
    """
    pedestrians_folder = "./Assets/Pedestrians/"
    frame_pattern = f"{pedestrians_folder}frame_{frame_number}_human_*.obj"
    pedestrian_files = glob.glob(frame_pattern)

    for pedestrian_file in pedestrian_files:
        # Import the pedestrian object
        bpy.ops.wm.import_scene.obj(filepath=pedestrian_file)

        # Apply transformations to the imported pedestrian objects
        for obj in bpy.context.selected_objects:
            obj.location.x += 1.5
            obj.location.y += 1.5
            obj.location.z += 1.5
            obj.rotation_euler.y += 3.14159  # 180 degrees in radians
            obj.rotation_euler.z += 3.14159  # 180 degrees in radians


def render_scene(data: List[dict], render_output_dir: str, VIDEO_NUMBER: int) -> None:
    """
    Render the Blender scene based on the provided data and save the output images.

    Parameters
    ----------
    data : list of dict
        List of frame data containing object details.
    render_output_dir : str
        Directory to save the rendered images.
    VIDEO_NUMBER : int
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

        # Add a Sun light source to the scene
        sun = bpy.data.lights.new(name="Sun", type="SUN")
        sun_object = bpy.data.objects.new(name="Sun", object_data=sun)
        bpy.context.collection.objects.link(sun_object)
        sun_object.location = (0, 0, 200)  # Position the Sun above the scene
        sun.energy = 5  # Set the brightness of the Sun

        # Set the camera's location and rotation
        camera_location = (0.0, 0.2, 1.4)
        camera_rotation = (1.57, 0.0, 0.0)
        if "Camera" in bpy.data.objects:
            camera = bpy.data.objects["Camera"]
            camera.location = camera_location
            camera.rotation_euler = camera_rotation

        # Extract object details for the current frame
        for obj in frame_data["objects"]:
            x, y, z = (obj["position"]["x"] - 9) * 1.2, obj["position"]["y"] / 1.5, 0
            phi, theta, psi = (
                degrees(obj["rotation"]["x"]),
                degrees(obj["rotation"]["y"]),
                degrees(obj["rotation"]["z"]),
            )

            if obj["type"] in vehicle_filepaths:
                blend_filepath = vehicle_filepaths[obj["type"]]
            elif obj["type"] in traffic_item_filepaths:
                blend_filepath = traffic_item_filepaths[obj["type"]]
            else:
                continue
            spawn_objects(blend_filepath, (x, y, z), (phi, theta, psi))

        # Add pedestrians to the scene for the current frame
        add_pedestrians_to_scene(frame_data["frame"])

        # Set up compositing nodes for rendering
        setup_compositing_nodes()

        # Check if a stop sign is detected in the current frame
        stop_sign_detected = any(
            obj["type"] == "stop sign" for obj in frame_data["objects"]
        )

        # Check if a speed limit sign is detected in the current frame
        speed_limit_detected = any(
            obj["type"] == "speed limit" for obj in frame_data["objects"]
        )

        # Set the visibility of the stop sign and speed limit sign based on detection
        if stop_sign_detected:
            # Set the stop sign texture to be visible
            for node in bpy.context.scene.node_tree.nodes:
                if node.name == "AlphaOver_StopSign":
                    node.inputs[0].default_value = 1.0
        else:
            # Hide the stop sign texture
            for node in bpy.context.scene.node_tree.nodes:
                if node.name == "AlphaOver_StopSign":
                    node.inputs[0].default_value = 0.1

        if speed_limit_detected:
            # Set the speed limit texture to be visible
            for node in bpy.context.scene.node_tree.nodes:
                if node.name == "AlphaOver_SpeedLimit":
                    node.inputs[0].default_value = 1.0
        else:
            # Hide the speed limit texture
            for node in bpy.context.scene.node_tree.nodes:
                if node.name == "AlphaOver_SpeedLimit":
                    node.inputs[0].default_value = 0.1

        # Render the scene and save the image
        bpy.ops.render.render(write_still=True)
        image_name = f"{frame_data['frame']:06d}.png"
        image_filepath = os.path.join(render_output_dir, image_name)
        bpy.data.images["Render Result"].save_render(image_filepath)
        print(f"Rendered image {image_name}")

        if frame_data["frame"] == 1370:
            break

    # Generate a video from the rendered images using pyffmpeg
    output_video_path = (
        f"{FFMPEG_OUTPUT_BASE_DIR}/vid_{VIDEO_NUMBER}/blender_render_video.mp4"
    )
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    ff = FFmpeg()
    input_pattern = os.path.join(render_output_dir, "*.png")
    ff.options(
        f"-y -framerate 30 -pattern_type glob -i {input_pattern} -c:v libx264 -r 30 -pix_fmt yuv420p {output_video_path}"
    )
    print(f"Blender render video saved at {output_video_path}")


def main() -> None:
    """
    Main function to load data, render the scene, and generate the video.

    Returns
    -------
    None
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Render 3D scenes in Blender.")
    parser.add_argument(
        "--video", type=int, required=True, help="Video number to process."
    )
    args = parser.parse_args()

    # Assign the video number from the argument
    VIDEO_NUMBER = args.video

    # Load the JSON file containing object details
    with open(SPAWN_JSON_PATH, "r") as file:
        data = json.load(file)

    # Define the output directory for renders based on the video number
    render_output_dir = f"{RENDER_OUTPUT_BASE_DIR}/vid_{VIDEO_NUMBER}"
    os.makedirs(render_output_dir, exist_ok=True)

    # Render the scene and generate the video
    render_scene(data, render_output_dir, VIDEO_NUMBER)


if __name__ == "__main__":
    main()
