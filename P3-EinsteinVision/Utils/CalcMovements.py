import numpy as np
import cv2
from numpy.distutils.system_info import x11_info
from ptlflow.utils import flow_utils


def _spacial_scale_flow(flow: np.array) -> np.array:
    """
    Scale the flow based on the distance of the object pixels from the center of the image
    Essentially, this is just going to be a giant radial gradient
    :param flow: The flow output in the shape of (H, W, 2)
    :return: a weighted and scaled flow
    """
    h, w = flow.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Create a meshgrid of the image coordinates
    x = np.arange(w)
    y = np.arange(h)

    X, Y = np.meshgrid(x, y)
    # Calculate the distance from the center of the image
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    # use these in a radial gradient
    # Normalize the distance to be between 0 and 1 and then add 1 to keep the image from going to 0
    dist = dist / np.max(dist) + 1
    # Scale the flow by the distance
    flow_scaled = flow * dist

    return flow_scaled

def approximate_average_flow(flow: np.array, movings: list[np.array]) -> np.array:
    """
    Calculate the approximate global flow of the scene. I.e. the average
    flow of the scene without any objects that may be moving
    :param flow: the optical flow between two images in the shape of (H, W, 2)
    :param movings: the list of movable objects in the scene (cars, trucks, etc.)
    :return: the average flow of the scene in the shape of (2,)
    """
    #Convert the depth map to cartesian coordinates
    flow_scaled = _spacial_scale_flow(flow)
    for mask in movings:
        flow_scaled[mask] = 0
    mean_flow_xy = np.mean(flow_scaled, axis=(0, 1))
    return mean_flow_xy

def predict_parked_objects(flow: np.array, depth1: np.array, depth2: np.array, boxes: np.array, thresh: float = 5) -> list[np.array]:
    """
    Predict the objects that are parked in the scene. This is done by
    calculating the flow of the objects in the scene and then using the
    depth maps to determine if the objects are moving or not.
    :param flow: the optical flow between two images in the shape of (H, W, 2)
    :param depth1: the depth map of image 1 in the shape of (H, W)
    :param depth2: the depth map of image 2 in the shape of (H, W)
    :param boxes: the list of bounding boxes for the objects in the scene
    :return: a dict of ID to boolean values indicating if the object is moving or not
        - True if the object is moving
        - False if the object is not moving
    """
    depth_flow = depth2 - depth1

    # Convert the depth map to cartesian coordinates
    # Get the average flow of the scene
    avg_flow = approximate_average_flow(flow, boxes)
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    classifications = dict()
    for i, box in enumerate(boxes):
        curr_box = box.astype(int)
        # Get the flow for the box
        curr_flow = flow[y1:y2, x1:x2]
        mean_flow = np.mean(curr_flow, axis=(0, 1))
        depth_flow = depth_flow[y1:y2, x1:x2]

        # Get the depth for the box
        depth_box = depth_flow[y1:y2, x1:x2]
        mean_depth = np.mean(depth_box)

        # Threshold the flow based on the average flow of the scene in pixels
        if np.linalg.norm(mean_flow - avg_flow) < thresh:
            classifications[i] = True
        # threshold the depth to be 1m
        elif mean_depth < 1:
            classifications[i] = True
        else:
            classifications[i] = False
    return boxes


############################################################################################################
######################## This was Nikesh messing around and it kinda maybe could work#######################
############################################################################################################



############################################################################################################
######################## This was Nikesh messing around and it kinda maybe could work#######################
############################################################################################################


def calc_movement(img1:np.array, img2:np.array, boxes:np.array, flow:np.array, fundamental: np.array):
    """
    Convert depth and uv coordinates to cartesian coordinates.
    K is the camera intrinsic matrix.
    depth is the depth map.
    uv is the uv coordinates.
    Returns the cartesian coordinates in the form of (x, y, z) and the distance to the object.
    """
    bx1, by1, bx2, by2 = boxes[0], boxes[1], boxes[2], boxes[3]
    flow_mask = flow[by1:by2, bx1:bx2]

    corners = cv2.goodFeaturesToTrack(img1[by1:by2, bx1:bx2], maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is None:
        return False
    points = np.int32(corners).reshape(-1, 2)
    h, w = img1.shape[:2]
    distances = []
    for point in points:
        x, y = point[0], point[1]
        # Get the flow for the point
        predicted_flow = flow_mask[y, x]
    bx1, by1, bx2, by2 = boxes[0], boxes[1], boxes[2], boxes[3]
    flow_mask = flow[by1:by2, bx1:bx2]

    corners = cv2.goodFeaturesToTrack(img1[by1:by2, bx1:bx2], maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is None:
        return False
    points = np.int32(corners).reshape(-1, 2)
    h, w = img1.shape[:2]
    distances = []
    for point in points:
        x, y = point[0], point[1]
        # Get the flow for the point
        predicted_flow = flow_mask[y, x]

        img1_homography = np.array([[x], [y], [1]])
        epipolar = fundamental @ img1_homography
        xy_prime = calc_epipolar_line(epipolar)

        if xy_prime is None:
            return False
        x_expected, y_expected = xy_prime[0], xy_prime[1]
        if x_expected is not None and 0 <= x_expected < w and 0 <= y_expected < h:
            actual_displacement = [img2[y_expected, x_expected] - img2[y, x]]

            # Sampson distance calculation
            sampson_distance = np.linalg.norm(predicted_flow - actual_displacement) ** 2 / (
                        predicted_flow[0] ** 2)
            distances.append(sampson_distance)

            # Classification based on average Sampson distance and threshold
        threshold = 2  # Adjust this based on your application and expected flow values
        # print(len(sampson_distances))
        # print("====================================")
        # print("Min Sampson Distance: ", np.min(sampson_distances))
        # print("Max Sampson Distance: ", np.max(sampson_distances))
        # print("Average Sampson Distance: ", np.mean(sampson_distances))
        # print("====================================")
        if len(distances) == 0:
            return True

        avg_sampson_distance = np.mean(distances)
        # True if moving
        # False if not moving
        return avg_sampson_distance < threshold



def calc_epipolar_line(epipolar: np.array, img_sz:tuple[int, int]=(1280,960)) -> np.array | None:
        """
        Compute intersection point of epipolar line with image boundaries.
        """
        x, y, z = epipolar[0], epipolar[1], epipolar[2]
        h, w = img_sz
        if z != 0:
            x_prime = 0
            y_prime = int((-x * x_prime - z) / y)
            if 0 <= y_prime < h:
                return x_prime, y_prime
            else:
                y_prime = h - 1
                x_prime = int((-y * y_prime - z) / x)
                if 0 <= x_prime < w:
                    return x_prime, y_prime
        return None


def _calc_fundamental(img1: np.array, img2: np.array) -> np.array:
    """
    Calculates a mask of the Sampson Distance on the optical flow between two images.
    :param img1: the base image in the shape of (H, W, 3)
    :param img2: the second frame in the shape of (H, W, 3)

    :return:
    """
    sift = cv2.SIFT.create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts2).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)
    return cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)




def _convert_to_cartesian(K, depth_value, u, v) -> np.ndarray:
    """
    Convert depth and uv coordinates to cartesian coordinates.
    K is the camera intrinsic matrix.
    depth is the depth map.
    uv is the uv coordinates.
    Returns the cartesian coordinates in the form of (x, y, z) and the distance to the object.
    """
    # Get the camera intrinsic matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Convert to cartesian coordinates
    # x is side to side
    # y is up and down
    # z is depth
    x =  (u-cx)*depth_value/fx
    y = (v-cy)*depth_value/fy
    z = depth_value
    dist = np.sqrt(x**2 + y**2 + z**2)
    return np.stack([x, y, z, dist], axis=-1)


