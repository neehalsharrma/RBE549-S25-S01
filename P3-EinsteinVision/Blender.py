import bpy
import json
import os


# Function to spawn objects from a .blend file at a specified location and rotation
def spawn_objects(filepath, location, rotation):
    # Load objects from the specified .blend file
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        data_to.objects = data_from.objects
    spawned_objects = []
    for obj in data_to.objects:
        if obj.parent is None:  # Only consider top-level objects
            new_obj = obj.copy()  # Create a copy of the object
            new_obj.data = obj.data.copy()  # Copy the object's data
            new_obj.animation_data_clear()  # Clear any animation data
            new_obj.location = location  # Set the object's location
            new_obj.rotation_euler = rotation  # Set the object's rotation
            new_obj.scale = (0.02, 0.02, 0.02)  # Scale the object
            bpy.context.collection.objects.link(new_obj)  # Link the object to the scene
            spawned_objects.append(new_obj)
            # Iterate through child objects and link them
            for child_obj in obj.children:
                new_child_obj = child_obj.copy()
                new_child_obj.data = child_obj.data.copy()
                new_child_obj.animation_data_clear()
                new_child_obj.parent = new_obj  # Set the parent-child relationship
                bpy.context.collection.objects.link(new_child_obj)
                spawned_objects.append(new_child_obj)


# Delete all objects in the scene except for the camera
for obj in bpy.context.scene.objects:
    if obj.type != "CAMERA":
        bpy.data.objects.remove(obj, do_unlink=True)

# Path to the .blend file containing the 3D models
blend_filepath = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/SedanAndHatchback.blend"
# Uncomment the following line to use a different .blend file
# blend_filepath = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/Assets/Vehicles/Truck.blend"

# Path to the JSON file containing object spawn data
file_path = "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/spawn.json"

# Load the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

# Initialize lists to store object details for each frame
frames = []  # Frame data
types = []  # Object types
positions = []  # Object positions
rotations = []  # Object rotations
scales = []  # Object scales

# Spawn a car object at the origin with a specific rotation
spawn_objects(blend_filepath, (0, 0, 0), (0, 0, 3.14))

# Set the camera's location and rotation
camera_location = (0.0, 0.2, 1.4)
camera_rotation = (1.57, 0.0, 0.0)

# Check if a camera exists in the scene and set its properties
if "Camera" in bpy.data.objects:
    camera = bpy.data.objects["Camera"]
camera.location = camera_location
camera.rotation_euler = camera_rotation

# Iterate through the JSON data to extract object details
for frame_data in data:
    for obj in frame_data["objects"]:
        types.append(obj["type"])
        positions.append(obj["position"])
        rotations.append(obj["rotation"])
        scales.append(obj["scale"])
    break  # Process only the first frame

# Spawn objects based on the extracted positions and rotations
for i in range(len(positions)):
    x, y, z = (positions[i]["x"], positions[i]["y"], 0)  # Extract position
    phi, theta, psi = (
        rotations[i]["x"],
        rotations[i]["y"],
        rotations[i]["z"],
    )  # Extract rotation
    spawn_objects(blend_filepath, (x, y, z), (phi, theta, psi))

# Render the scene and save the image
bpy.ops.render.render(write_still=True)
print(f"Rendered image {i:06d}.png")
image_name = f"{i:06d}.png"
image_filepath = os.path.join(
    "/home/nasharrma/RBE549-S25-S01/P3-EinsteinVision/blender_py/rendered_images",
    image_name,
)
bpy.data.images["Render Result"].save_render(image_filepath)
