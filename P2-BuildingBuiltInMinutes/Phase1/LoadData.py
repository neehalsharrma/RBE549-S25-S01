import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_data(img: int, data_path: str = '../P2Data/', num_images: int = 5) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Load the data from the given path and return the data as a tuple.
    @ data_path: The path to the data.
    @ img: The image to load.
    @ num_images: The number of images to be used in the SfM matching
    @ return: A tuple containing the number of matches, the data from the file, and a list containing the images.
    Format for matches:
    Each Row: (the number of matches for the jth feature)
              (Red Value) (Green Value) (Blue Value)
              (u_current image) (v_current image)
              (image id) (u_{image_id image}) (v_{image_id_image})
              (image id) (u_{image_id_image}) (v_{image id image}) …
    """
    # Load the data from the file.
    matching_file = data_path + 'matching' + str(img) + '.txt'
    header_data = 6
    with open(matching_file, 'r') as file:
        lines = file.readlines()
        # Extract the number of features
        n_features = int(lines[0].split(":")[1].strip())
        num_lines = len(lines) - 1
        # Process the feature data into a NumPy array
        matches = np.zeros((num_lines, header_data + (num_images - 1) * 3), dtype=np.float32)
        for i, line in enumerate(lines[1:]):
            values = list(map(float, line.split()))
            matches[i, :len(values)] = np.array(values)

    img_path = data_path + str(img) + '.png'

    # Load the images
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return n_features, matches, img


def show_features(points, img1):
    img1_features = img1.copy()
    for i in range(points.shape[0]):
        y, x = points[i, 4:6]
        cv2.circle(img1_features, (int(x), int(y)), 5, (0, 255, 0), -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img1_features)
    plt.axis('off')
    plt.show()


def show_matches(image1: int, image2: int, data_path: str = '../P2Data/'):
    n_features_one, matches_one, img1 = load_data(image1)
    n_features_two, matches_two, img2 = load_data(image2)
    spacer = 6  # The number of columns in the matches array
    img = np.concatenate((img1, img2), axis=1)
    for i in range(len(matches_one)):
        x1, y1 = matches_one[i, 4:6]
        num_matches = int(matches_one[i, 0])
        for j in range(num_matches - 1):
            img_id = int(matches_one[i, spacer + j * 3])
            if img_id != image2:
                continue
            x2, y2 = matches_one[i, 7 + j * 3:9 + j * 3]


            cv2.circle(img, (int(x1), int(y1)), 3, (0, 0, 255), 2)
            cv2.circle(img, (int(x2) + img1.shape[1], int(y2)), 3, (0, 0, 255), 2)
            cv2.line(img, (int(x1), int(y1)), (int(x2) + img1.shape[1], int(y2)), (255, 0, 0), 2)
    plt.figure(figsize=(10, 10))

    plt.imshow(img)
    plt.axis('off')
    plt.show()
