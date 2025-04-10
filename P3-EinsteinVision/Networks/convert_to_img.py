import numpy as np
import cv2
import os
import matplotlib
files = os.listdir("vid_9")
cmap = matplotlib.colormaps.get_cmap('viridis')
depth_dir = "vid_9_depth_maps"
os.makedirs(depth_dir, exist_ok=True)

for frame_idx in range(len(files)):
    frame_idx *= 5
    depth = np.load(os.path.join("vid_9", f'depth_{frame_idx}.npy'))
    # Convert the array to an image
    # Used to check for differences between the two methods of saving the depth map
    # np.save(os.path.join(depth_dir, f"depth_frame_16_{frame_idx}.npy"), depth.astype(np.uint16))
    # Colorize the depth map
    colorized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    colorized = colorized.astype(np.uint8)
    colorized = (cmap(colorized)[:, :, :3] * 255).astype(np.uint8)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        os.path.join(depth_dir, f"depth_frame_color_{frame_idx}.png"),
        colorized)
