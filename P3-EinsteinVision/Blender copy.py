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
from typing import List, Tuple

import bpy

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
    "stop sign": "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/StopSign.blend",
}

# Global constants
SPAWN_JSON_PATH = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/spawn.json"
WRAPPER_SCRIPT_PATH = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Wrapper.py"
RENDER_OUTPUT_BASE_DIR = (
    "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Results/Renders"
)
STOP_SIGN_TEXTURE_PATH = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/StopSignImage.png"  # Path to stop sign texture


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


def apply_texture_to_object(image_path, obj):
    """
    Apply a texture to a given Blender object using an image file.

    This function creates a new material, enables nodes for the material,
    and assigns an image texture to the material. The texture is then linked
    to the Principled BSDF shader's Base Color input. Finally, the material
    is assigned to the specified object.

    Parameters
    ----------
    image_path : str
        The file path to the image to be used as the texture.
    obj : bpy.types.Object
        The Blender object to which the texture will be applied.

    Returns
    -------
    None
    """
    # Create a new material and enable the use of nodes
    material = bpy.data.materials.new(name="ImageTexture")
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Create an Image Texture node and load the image
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(image_path)

    # Get the Principled BSDF node and link the texture to its Base Color input
    bsdf_node = nodes.get("Principled BSDF")
    material.node_tree.links.new(
        tex_node.outputs["Color"], bsdf_node.inputs["Base Color"]
    )

    # Assign the material to the object
    if obj.data.materials:
        # Replace the first material if the object already has materials
        obj.data.materials[0] = material
    else:
        # Add the new material if the object has no materials
        obj.data.materials.append(material)


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

        # Check if a stop sign is detected in the current frame
        stop_sign_detected = any(
            obj["type"] == "stop sign" for obj in frame_data["objects"]
        )

        if stop_sign_detected:
            # Add image alert on screen if a stop sign is detected
            # Enable compositing
            bpy.context.scene.use_nodes = True
            nodes = bpy.context.scene.node_tree.nodes
            links = bpy.context.scene.node_tree.links

            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)

            # Add render layer node
            render_layer_node = nodes.new(type="CompositorNodeRLayers")
            render_layer_node.location = (0, 0)

            # Add image node
            image_node = nodes.new(type="CompositorNodeImage")
            image_node.location = (200, 200)
            image_node.image = bpy.data.images.load(STOP_SIGN_TEXTURE_PATH)

            # Add transform node (NEW: Controls scale/position)
            transform_node = nodes.new(type="CompositorNodeTransform")
            transform_node.location = (400, 200)
            transform_node.inputs["Scale"].default_value = 0.2
            transform_node.inputs["Y"].default_value = 400

            # Add alpha over node
            alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
            alpha_over_node.location = (600, 0)

            # Add composite output node
            composite_node = nodes.new(type="CompositorNodeComposite")
            composite_node.location = (800, 0)

            # Link nodes
            links.new(render_layer_node.outputs["Image"], alpha_over_node.inputs[1])
            links.new(
                image_node.outputs["Image"], transform_node.inputs["Image"]
            )  # Image → Transform
            links.new(
                transform_node.outputs["Image"], alpha_over_node.inputs[2]
            )  # Transform → Alpha Over
            links.new(alpha_over_node.outputs["Image"], composite_node.inputs["Image"])

        # Check if a speed limit sign is detected in the current frame
        speed_limit_detected = any(
            obj["type"] == "speed limit" for obj in frame_data["objects"]
        )

        if speed_limit_detected:
            # Add image alert on screen if a speed limit sign is detected
            # Enable compositing
            bpy.context.scene.use_nodes = True
            nodes = bpy.context.scene.node_tree.nodes
            links = bpy.context.scene.node_tree.links

            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)

            # Add render layer node
            render_layer_node = nodes.new(type="CompositorNodeRLayers")
            render_layer_node.location = (0, 0)

            # Add image node
            image_node = nodes.new(type="CompositorNodeImage")
            image_node.location = (200, 200)
            image_node.image = bpy.data.images.load(SPEED_LIMIT_TEXTURE_PATH)

            # Add transform node (NEW: Controls scale/position)
            transform_node = nodes.new(type="CompositorNodeTransform")
            transform_node.location = (400, 200)
            transform_node.inputs["Scale"].default_value = 0.2
            transform_node.inputs["X"].default_value = 100
            transform_node.inputs["Y"].default_value = 400

            # Add alpha over node
            alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
            alpha_over_node.location = (600, 0)

            # Add composite output node
            composite_node = nodes.new(type="CompositorNodeComposite")
            composite_node.location = (800, 0)

            # Link nodes
            links.new(render_layer_node.outputs["Image"], alpha_over_node.inputs[1])
            links.new(
                image_node.outputs["Image"], transform_node.inputs["Image"]
            )  # Image → Transform
            links.new(
                transform_node.outputs["Image"], alpha_over_node.inputs[2]
            )  # Transform → Alpha Over
            links.new(alpha_over_node.outputs["Image"], composite_node.inputs["Image"])

        # Render the scene and save the image
        bpy.ops.render.render(write_still=True)
        image_name = f"{frame_data['frame']:06d}.png"
        image_filepath = os.path.join(render_output_dir, image_name)
        bpy.data.images["Render Result"].save_render(image_filepath)
        print(f"Rendered image {image_name}")

        # Reset compositing and image overlays for the next render
        scene = bpy.context.scene
        scene.use_nodes = True
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links

        # Clear all newly added nodes in the compositor
        for node in list(nodes):
            if node.name not in {"Render Layers", "Composite"}:
                nodes.remove(node)

        # Ensure the remaining nodes are linked together
        render_layer_node = nodes.get("Render Layers")
        composite_node = nodes.get("Composite")
        if render_layer_node and composite_node:
            # Clear existing links
            for link in list(links):
                links.remove(link)
            # Link Render Layers to Composite
            links.new(
                render_layer_node.outputs["Image"], composite_node.inputs["Image"]
            )

        if frame_data["frame"] == 10:
            break


def main() -> None:
    """
    Main function to load data, render the scene, and generate the video.

    Returns
    -------
    None
    """
    # Assign the video number from the argument
    VIDEO_NUMBER = 8

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
