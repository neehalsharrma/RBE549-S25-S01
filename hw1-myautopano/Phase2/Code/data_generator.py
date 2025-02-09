from typing import List, Optional  # For type hinting

import concurrent.futures  # For parallel processing
import os  # For file and directory operations
import random  # For generating random numbers
import re  # For regular expressions
import time  # For measuring time

import cv2  # OpenCV library for image processing
import numpy as np  # For numerical operations

# Define relative path and label file name as run from the repository level
relative_path = "hw1-myautopano/Phase2/Data/Train/"
label_file_name = "../TrainLabels.csv"
time_read_write = 0  # Initialize time for read/write operations
total_time = 0  # Initialize total time


def setup(relative_path: str) -> None:
    """
    Create directories for PA and PB if they do not exist.

    Parameters:
        relative_path (str): The relative path where the directories PA and PB will be created.

    Returns:
        None
    """
    if not os.path.exists(relative_path + "PA"):
        os.makedirs(relative_path + "PA")
    if not os.path.exists(relative_path + "PB"):
        os.makedirs(relative_path + "PB")


def get_image_list(image_path: str) -> List[str]:
    """
    Get a sorted list of image files from the given directory.

    Parameters:
        image_path (str): The path to an image file within the target directory.

    Returns:
        list: A sorted list of image file names (str) with ".jpg" extension from the directory.
              The files are sorted based on the numerical value extracted from their names.
    """
    image_directory = os.path.dirname(image_path)
    all_files = os.listdir(image_directory)  # List all files in the directory
    image_files = [
        f for f in all_files if f.endswith(".jpg")
    ]  # Filter out the image files
    sorted_files = sorted(
        image_files, key=lambda x: int(re.search(r"\d+", x).group())
    )  # Sort files by number
    return sorted_files


def get_batch(image_list: List[str], batchNum: int, size: int) -> List[str]:
    """
    Retrieves a batch of images from the provided image list.

    Parameters:
        image_list (list): A list of images.
        batchNum (int): The batch number to retrieve.
        size (int): The size of each batch.

    Returns:
        list: A sublist of images corresponding to the specified batch.
    """

    tot = len(image_list)
    return image_list[batchNum * size : min((batchNum + 1) * size, tot)]


def get_images(image_list: List[str]) -> List[np.ndarray]:
    """Read images from the list and return them."""
    global time_read_write
    images = []
    start_time = time.time()
    for image_file in image_list:
        images.append(cv2.imread(relative_path + image_file))
    end_time = time.time()
    time_read_write += end_time - start_time  # Update read/write time
    return images


def get_perturbvals(p: int) -> List[int]:
    """
    Generate a list of 8 random integers within the range [-p, p].

    Parameters:
        p (int): The range limit for the random integers. The integers will be between -p and p inclusive.

    Returns:
        list: A list containing 8 random integers within the specified range.
    """

    return [random.randint(-p, p) for i in range(8)]


def get_coordvals(p: int, h: int, w: int, patch_size: int) -> Optional[List[int]]:
    """
    Generate random coordinates for patches.

    Parameters:
        p (int): Padding value to ensure patches are within bounds.
        h (int): Height of the image.
        w (int): Width of the image.
        patch_size (int): Size of the patch to be generated.

    Returns:
        list: A list containing the coordinates [x1, x2, y1, y2] of the patch.
              Returns None if the patch cannot be generated within the given constraints.
    """
    limitx = w - patch_size - (2 * p)
    limity = h - (2 * p) - patch_size
    if limitx < 2 * p or limity < 2 * p:
        return None
    x1 = random.randint(2 * p, limitx)
    y1 = random.randint(2 * p, limity)
    l = [x1, x1 + patch_size, y1, y1 + patch_size]
    return l


def get_randompatches(
    img: np.ndarray, p: int, patch_size: int, iname: str, patches_per_image: int
) -> List[np.ndarray]:
    """
    Generate random patches from the image.

    Parameters:
        img (numpy.ndarray): The input image from which patches are to be generated.
        p (int): Perturbation value to generate random coordinates.
        patch_size (int): The size of the patches to be extracted.
        iname (str): The base name for saving the patches.
        patches_per_image (int): The number of patches to generate from the image.

    Returns:
        list: A list of tildaH values representing the differences between the original and perturbed coordinates.
    """
    h, w, _ = img.shape
    aHab = []  # List of tildaH values
    for i in range(patches_per_image):
        coords = get_coordvals(p, h, w, patch_size)
        while coords is None:
            coords = get_coordvals(p * 2 // 3, h, w, patch_size)
        w1, w4, h1, h4 = coords
        ph1, ph2, ph3, ph4, pw1, pw2, pw3, pw4 = get_perturbvals(p)
        ca = np.float32([[w1, h1], [w1, h4], [w4, h1], [w4, h4]])
        cb = np.float32(
            [
                [w1 + pw1, h1 + ph1],
                [w1 + pw2, h4 + ph2],
                [w4 + pw3, h1 + ph3],
                [w4 + pw4, h4 + ph4],
            ]
        )
        Hab = cv2.getPerspectiveTransform(ca, cb)
        Hba = np.linalg.inv(Hab)
        Pa = img[h1:h4, w1:w4]
        transformed_img = cv2.warpPerspective(img, Hba, (w, h))
        Pb = transformed_img[h1:h4, w1:w4]
        aHab.append(np.subtract(cb, ca).flatten())
        save_randpatches(Pa, Pb, iname + "_" + str(i + 1))
    return aHab


def save_randpatches(Pa: np.ndarray, Pb: np.ndarray, iname: str) -> None:
    """
    Save random patches Pa and Pb as images with the given name.
    This function saves the provided image patches Pa and Pb to the disk with filenames
    derived from the provided iname. The images are saved in directories "PA" and "PB"
    respectively, with the suffix "A.jpg" and "B.jpg" appended to the iname.

    Parameters:
        Pa (numpy.ndarray): The first image patch to be saved.
        Pb (numpy.ndarray): The second image patch to be saved.
        iname (str): The base name for the saved image files.

    Returns:
        None
    """

    global time_read_write
    start_time = time.time()
    cv2.imwrite(str(relative_path + "PA/" + iname + "A.jpg"), Pa)
    cv2.imwrite(str(relative_path + "PB/" + iname + "B.jpg"), Pb)
    end_time = time.time()
    time_read_write += end_time - start_time  # Update read/write time


def generate_images_batch(
    p: float,
    batch_images: List[np.ndarray],
    batch_num: int,
    batch_size: int,
    patch_size: int,
    patches_per_image: int,
) -> List[np.ndarray]:
    """
    Generates a batch of image patches with random transformations.

    Parameters:
        p (float): Probability parameter for random patch generation.
        batch_images (list): List of images to generate patches from.
        batch_num (int): The current batch number.
        batch_size (int): The number of images in each batch.
        patch_size (int): The size of each patch.
        patches_per_image (int): The number of patches to generate per image.

    Returns:
        list: A list of transformed patches from the batch of images.
    """
    i = 0
    transformation_list = []
    for image in batch_images:
        i += 1
        aHab = get_randompatches(
            image, p, patch_size, str(batch_num * batch_size + i), patches_per_image
        )
        transformation_list.extend(aHab)
    return transformation_list


def save_list_to_file(data: List[List[float]], filename: str) -> None:
    """
    Saves a list of lists to a file, with each inner list written as a comma-separated line.

    Parameters:
        data (list of list of any): The data to be written to the file. Each inner list represents a line.
        filename (str): The name of the file where the data will be saved.

    Returns:
        None
    """

    with open(filename, "w") as file:
        for line in data:
            file.write(",".join(map(str, line)) + "\n")


def generate_images(
    p: float,
    relative_path: str,
    batch_size: int,
    patch_size: int,
    patches_per_image: int,
) -> None:
    """
    Generate transformed images in batches and save the transformation list to a file.

    Parameters:
        p (float): Parameter for image transformation.
        relative_path (str): Path to the directory containing images.
        batch_size (int): Number of images to process in each batch.
        patch_size (int): Size of the patches to generate from each image.
        patches_per_image (int): Number of patches to generate from each image.

    Returns:
        None
    """

    image_list = get_image_list(relative_path)
    transformation_list = []

    iterations = (len(image_list) + batch_size - 1) // batch_size  # Ceiling division

    def process_batch(i: int) -> List[np.ndarray]:
        """
        Processes a batch of images and generates a batch of image patches.

        Parameters:
            i (int): The index of the current batch.

        Returns:
            list: A list of generated image patches for the current batch.
        """

        print("On batch- ", i)
        batch_list = get_batch(image_list, i, batch_size)
        batch_images = get_images(batch_list)
        return generate_images_batch(
            p, batch_images, i, batch_size, patch_size, patches_per_image
        )

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, range(iterations)))

    # Flatten the list of results
    for result in results:
        transformation_list.extend(result)

    save_list_to_file(transformation_list, relative_path + label_file_name)


# Main execution
start = time.time()
setup(relative_path)
# p, image path, batch_size, patch_size, patches_per_image
generate_images(30, relative_path, 50, 128, 2)
end = time.time()
total_time = end - start

print("Total time- ", total_time, "    time_read_write- ", time_read_write)
