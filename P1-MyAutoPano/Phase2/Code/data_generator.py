"""
data_generator.py

This module generates image patches for training or testing purposes. It includes functions
to read images, generate random patches, save patches to disk, and manage multithreading
for efficient processing.

Usage:
    python data_generator.py --train  # For training data generation
    python data_generator.py --test   # For testing data generation
    (Run at the package level)

Command-line arguments:
    --train: Set this flag for training data generation
    --test: Set this flag for testing data generation
    --workers: Number of workers for multithreading (default: min(8, number of logical CPUs))

Functions:
    setup(relative_path: str) -> None
    get_list_images(image_path: str) -> List[str]
    get_batch(image_list: List[str], batch_num: int, size: int) -> List[str]
    get_images(image_list: List[str]) -> List[np.ndarray]
    get_perturb_vals(p: int) -> List[int]
    get_coord_vals(p: int, h: int, w: int, patch_size: int) -> Optional[List[int]]
    get_random_patches(img: np.ndarray, p: int, patch_size: int, iname: str, patches_per_image: int) -> Tuple[List[np.ndarray], List[str], List[str]]
    save_random_patches(pa: np.ndarray, pb: np.ndarray, file_a: str, file_b: str) -> None
    generate_images_batch(p: int, batch_images: List[np.ndarray], batch_num: int, batch_size: int, patch_size: int, patches_per_image: int) -> Tuple[List[np.ndarray], List[str], List[str]]
    save_labels_to_file(data: List[List[float]], filename: str) -> None
    save_image_list_to_file(data: List[str], filename: str) -> None
    generate_images(p: int, relative_path: str, batch_size: int, patch_size: int, patches_per_image: int) -> None
"""

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import os  # OS module for file operations
import random  # Random module for generating random numbers
import time  # Time module for measuring execution time
import re  # Regular expressions for string operations
import concurrent.futures  # For multithreading
import psutil  # For system and process utilities
import argparse  # For command-line argument parsing
from typing import List, Tuple, Optional  # For type hinting

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Generate image patches for training or testing."
)
parser.add_argument(
    "--train", action="store_true", help="Set this flag for training data generation"
)
parser.add_argument(
    "--test", action="store_true", help="Set this flag for testing data generation"
)
parser.add_argument(
    "--workers",
    type=int,
    default=min(8, psutil.cpu_count(logical=False)),
    help="Number of workers for multithreading",
)
args = parser.parse_args()

# Set paths based on whether training or testing
if args.train:
    # For training
    RELATIVE_PATH = "Data/Train/"
    LABEL_FILE_NAME = "Code/TxtFiles/TrainLabels.csv"
    PA_PATHS = "Code/TxtFiles/DirNamesTrainPA.txt"
    PB_PATHS = "Code/TxtFiles/DirNamesTrainPB.txt"
elif args.test:
    # For testing
    RELATIVE_PATH = "Data/Val/"
    LABEL_FILE_NAME = "Code/TxtFiles/TestLabels.csv"
    PA_PATHS = "Code/TxtFiles/DirNamesTestPA.txt"
    PB_PATHS = "Code/TxtFiles/DirNamesTestPB.txt"
else:
    raise ValueError("Please specify either --train or --test")


# Initialize timing variables
TIME_READ_WRITE = 0
TOTAL_TIME = 0


def setup(relative_path: str) -> None:
    """
    Create directories for PA and PB if they do not exist.

    Args:
        relative_path (str): The base path where directories will be created.
    """
    if not os.path.exists(os.path.join(relative_path, "PA")):
        os.makedirs(os.path.join(relative_path, "PA"))
        print("Created PA directory", os.path.join(relative_path, "PA"))
    if not os.path.exists(os.path.join(relative_path, "PB")):
        os.makedirs(os.path.join(relative_path, "PB"))
        print("Created PB directory", os.path.join(relative_path, "PB"))


def get_list_images(image_path: str) -> List[str]:
    """
    Get a sorted list of image files in the specified directory.

    Args:
        image_path (str): The path to the directory containing images.

    Returns:
        List[str]: A sorted list of image filenames.
    """
    print("Looking for images in -", image_path)
    image_directory = os.path.dirname(image_path)

    # List all files in the directory
    all_files = os.listdir(image_directory)
    print("Total files found- ", len(all_files))
    # Filter out the image files (assuming they are .jpg files)
    image_files = [f for f in all_files if f.endswith(".jpg")]
    sorted_files = sorted(image_files, key=lambda x: int(re.search(r"\d+", x).group()))
    print("Total image files found- ", len(sorted_files))
    return sorted_files


def get_batch(image_list: List[str], batch_num: int, size: int) -> List[str]:
    """
    Get a batch of images from the list.

    Args:
        image_list (List[str]): The list of image filenames.
        batch_num (int): The batch number.
        size (int): The size of the batch.

    Returns:
        List[str]: A list of image filenames for the batch.
    """
    total = len(image_list)
    # Calculate the start and end indices for the batch
    start_index = batch_num * size
    end_index = min((batch_num + 1) * size, total)
    # Return the sublist of image filenames for the batch
    return image_list[start_index:end_index]


def get_images(image_list: List[str]) -> List[np.ndarray]:
    """
    Read images from the list of filenames.

    Args:
        image_list (List[str]): The list of image filenames.

    Returns:
        List[np.ndarray]: A list of images.
    """
    global TIME_READ_WRITE
    images = []
    start_time = time.time()
    # Read each image file and append it to the images list
    for image_file in image_list:
        images.append(cv2.imread(RELATIVE_PATH + image_file))
    end_time = time.time()
    # Update the time spent on reading images
    TIME_READ_WRITE += end_time - start_time
    return images


def get_perturb_vals(p: int) -> List[int]:
    """
    Generate random perturbation values.

    Args:
        p (int): The range for perturbation values.

    Returns:
        List[int]: A list of random perturbation values.
    """
    return [random.randint(-p, p) for _ in range(8)]


def get_coord_vals(p: int, h: int, w: int, patch_size: int) -> Optional[List[int]]:
    """
    Generate random coordinates for patches.

    Args:
        p (int): The perturbation value.
        h (int): The height of the image.
        w (int): The width of the image.
        patch_size (int): The size of the patch.

    Returns:
        Optional[List[int]]: A list of coordinates or None if limits are exceeded.
    """
    # Calculate the limits for x and y coordinates
    limit_x = w - patch_size - (2 * p)
    limit_y = h - (2 * p) - patch_size
    
    # Check if the limits are valid
    if limit_x < 2 * p or limit_y < 2 * p:
        return None
    
    # Generate random coordinates within the limits
    x1 = random.randint(2 * p, limit_x)
    y1 = random.randint(2 * p, limit_y)
    
    # Return the coordinates as a list
    return [x1, x1 + patch_size, y1, y1 + patch_size]


def get_random_patches(
    img: np.ndarray, p: int, patch_size: int, iname: str, patches_per_image: int
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Generate random patches from an image.

    Args:
        img (np.ndarray): The input image.
        p (int): The perturbation value.
        patch_size (int): The size of the patch.
        iname (str): The image name.
        patches_per_image (int): The number of patches per image.

    Returns:
        Tuple[List[np.ndarray], List[str], List[str]]: A tuple containing the list of transformations, file A paths, and file B paths.
    """
    h, w, _ = img.shape  # Get the height and width of the image
    a_hab = []  # List of tildaH values
    file_a_paths = []  # List of file paths for patch A
    file_b_paths = []  # List of file paths for patch B

    for i in range(patches_per_image):
        coords = get_coord_vals(p, h, w, patch_size)  # Get random coordinates for patches
        max_iterations = 10  # Maximum number of iterations to find valid coordinates
        iterations = 0
        min_p = 1  # Minimum perturbation value

        # Try to find valid coordinates within the maximum number of iterations
        while coords is None and iterations < max_iterations and p > min_p:
            p = p // 2  # Reduce the perturbation value
            coords = get_coord_vals(p, h, w, patch_size)  # Get new coordinates
            iterations += 1

        if coords is None:
            raise ValueError("Unable to find valid coordinates for patches.")

        w1, w4, h1, h4 = coords  # Extract coordinates
        ph1, ph2, ph3, ph4, pw1, pw2, pw3, pw4 = get_perturb_vals(p)  # Get perturbation values

        # Define the coordinates for patch A and patch B
        coords_a = np.array([[w1, h1], [w1, h4], [w4, h1], [w4, h4]], dtype=np.float32)
        coords_b = np.array(
            [
                [w1 + pw1, h1 + ph1],
                [w1 + pw2, h4 + ph2],
                [w4 + pw3, h1 + ph3],
                [w4 + pw4, h4 + ph4],
            ]
        )

        hab = cv2.getPerspectiveTransform(coords_a, coords_b)  # Get perspective transform matrix
        hba = np.linalg.inv(hab)  # Get inverse perspective transform matrix
        pa = img[h1:h4, w1:w4]  # Extract patch A
        transformed_img = cv2.warpPerspective(
            img, hba, (w, h)
        )  # Apply inverse transform
        pb = transformed_img[h1:h4, w1:w4]  # Extract patch B

        a_hab.append(np.subtract(coords_b, coords_a).flatten())  # Calculate transformation

        # Define file paths for patch A and patch B
        file_a = str(
            os.path.join(
                RELATIVE_PATH, os.path.join("PA", iname + "_" + str(i + 1) + "A.jpg")
            )
        )
        file_b = str(
            os.path.join(
                RELATIVE_PATH, os.path.join("PB", iname + "_" + str(i + 1) + "B.jpg")
            )
        )

        file_a_paths.append(file_a)  # Add file path for patch A to the list
        file_b_paths.append(file_b)  # Add file path for patch B to the list

        save_random_patches(pa, pb, file_a, file_b)  # Save patches to disk

    return a_hab, file_a_paths, file_b_paths  # Return the transformations and file paths


def save_random_patches(
    pa: np.ndarray, pb: np.ndarray, file_a: str, file_b: str
) -> None:
    """
    Save random patches to disk.

    Args:
        pa (np.ndarray): The patch A image.
        pb (np.ndarray): The patch B image.
        file_a (str): The file path for patch A.
        file_b (str): The file path for patch B.
    """
    global TIME_READ_WRITE
    start_time = time.time()
    cv2.imwrite(file_a, pa)  # Save patch A
    cv2.imwrite(file_b, pb)  # Save patch B
    end_time = time.time()
    TIME_READ_WRITE += end_time - start_time  # Update the time spent on writing images


def generate_images_batch(
    p: int,
    batch_images: List[np.ndarray],
    batch_num: int,
    batch_size: int,
    patch_size: int,
    patches_per_image: int,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Generate a batch of images with random patches.

    Args:
        p (int): The perturbation value.
        batch_images (List[np.ndarray]): The list of batch images.
        batch_num (int): The batch number.
        batch_size (int): The size of the batch.
        patch_size (int): The size of the patch.
        patches_per_image (int): The number of patches per image.

    Returns:
        Tuple[List[np.ndarray], List[str], List[str]]: A tuple containing the list of transformations, file A paths, and file B paths.
    """
    i = 0
    transformation_list = []  # List to store transformations
    file_a_list = []  # List to store file paths for patch A
    file_b_list = []  # List to store file paths for patch B
    for image in batch_images:
        i += 1
        # Generate random patches for the image
        a_hab, file_as, file_bs = get_random_patches(
            image, p, patch_size, str(batch_num * batch_size + i), patches_per_image
        )
        transformation_list.extend(a_hab)  # Add transformations to the list
        file_a_list.extend(file_as)  # Add file paths for patch A to the list
        file_b_list.extend(file_bs)  # Add file paths for patch B to the list
    return transformation_list, file_a_list, file_b_list  # Return the results


def save_labels_to_file(data: List[List[float]], filename: str) -> None:
    """
    Save labels to a file.

    Args:
        data (List[List[float]]): The list of labels.
        filename (str): The file path to save the labels.
    """
    # Open the file in write mode with UTF-8 encoding
    with open(filename, "w", encoding="utf-8") as file:
        # Write each label list as a comma-separated string
        for line in data:
            file.write(",".join(map(str, line)) + "\n")


def save_image_list_to_file(data: List[str], filename: str) -> None:
    """
    Save image list to a file.

    Args:
        data (List[str]): The list of image file paths.
        filename (str): The file path to save the image list.
    """
    # Open the file in write mode with UTF-8 encoding
    with open(filename, "w", encoding="utf-8") as file:
        # Write each image file path on a new line
        for line in data:
            file.write(line + "\n")


def generate_images(
    p: int, relative_path: str, batch_size: int, patch_size: int, patches_per_image: int
) -> None:
    """
    Generate images with random patches.

    Args:
        p (int): The perturbation value.
        relative_path (str): The base path for saving images.
        batch_size (int): The size of the batch.
        patch_size (int): The size of the patch.
        patches_per_image (int): The number of patches per image.
    """
    # Get the list of image filenames
    image_list = get_list_images(relative_path)
    transformation_list = []  # List to store transformations
    file_a_list = []  # List to store file paths for patch A
    file_b_list = []  # List to store file paths for patch B

    # Calculate the number of iterations (batches)
    iterations = (len(image_list) + batch_size - 1) // batch_size  # Ceiling division
    print("Total iterations: ", iterations)

    def process_batch(i: int) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Process a single batch of images.

        Args:
            i (int): The batch index.

        Returns:
            Tuple[List[np.ndarray], List[str], List[str]]: A tuple containing the list of transformations, file A paths, and file B paths.
        """
        print("On batch- ", i)
        # Get the list of image filenames for the batch
        batch_list = get_batch(image_list, i, batch_size)
        # Read the images from the filenames
        batch_images = get_images(batch_list)
        # Generate random patches for the batch of images
        return generate_images_batch(
            p, batch_images, i, batch_size, patch_size, patches_per_image
        )

    # Use ThreadPoolExecutor for multithreading
    max_workers = args.workers  # Number of workers for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each batch in parallel
        results = list(executor.map(process_batch, range(iterations)))

    # Flatten the list of results
    for result in results:
        transformation_list.extend(result[0])
        file_a_list.extend(result[1])
        file_b_list.extend(result[2])

    # Save the file paths and transformations to disk
    save_image_list_to_file(file_a_list, PA_PATHS)
    save_image_list_to_file(file_b_list, PB_PATHS)
    save_labels_to_file(
        transformation_list, LABEL_FILE_NAME)


# Measure the total execution time
start = time.time()
# Setup the directories for saving patches
setup(RELATIVE_PATH)
# Generate images with random patches
generate_images(30, RELATIVE_PATH, 50, 128, 2)
end = time.time()
TOTAL_TIME = end - start
print(
    f"Total time: {TOTAL_TIME:.2f} seconds, Time spent on read/write: {TIME_READ_WRITE:.2f} seconds"
)
