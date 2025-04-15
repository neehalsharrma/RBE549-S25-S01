import numpy as np
import cv2


"""
Psuedocode for flow detections and movement calculations

Get the Z optical flow by using depth maps between images

Radially scale the optical flow based on the angle from the center of the image

"""


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

def approximate_z_flow(obj1: np.array, obj2: np.array) -> float:
    """
    Calculate the approximate Z flow between two objects.
    :param obj1: the depth map of the object in frame one
    :param obj2: the depth map of the object in frame two
    :return: the calculated Z flow as a single scalar value relative to the centroid
    """
    pass


def _calc_sampson_distance(img1: np.array, img2: np.array, flow: np.array) -> np.array:
    """
    Calculates a mask of the Sampson Distance on the optical flow between two images.
    :param img1: the base image in the shape of (H, W, 3)
    :param img2: the second frame in the shape of (H, W, 3)
    :param flow: the optical flow between the two images in the shape of (H, W, 2)
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

    fundamental = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)


