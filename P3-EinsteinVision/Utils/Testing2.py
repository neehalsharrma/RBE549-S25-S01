import cv2
import numpy as np


def nothing(x):
    pass


def create_trackbars(window_name):
    # Create trackbars for lower and upper threshold values for Y, Cr, Cb
    cv2.createTrackbar('Y min', window_name, 0, 255, nothing)
    cv2.createTrackbar('Y max', window_name, 255, 255, nothing)
    cv2.createTrackbar('Cr min', window_name, 0, 255, nothing)
    cv2.createTrackbar('Cr max', window_name, 255, 255, nothing)
    cv2.createTrackbar('Cb min', window_name, 0, 255, nothing)
    cv2.createTrackbar('Cb max', window_name, 255, 255, nothing)


def get_thresholds(window_name):
    y_min = cv2.getTrackbarPos('Y min', window_name)
    y_max = cv2.getTrackbarPos('Y max', window_name)
    cr_min = cv2.getTrackbarPos('Cr min', window_name)
    cr_max = cv2.getTrackbarPos('Cr max', window_name)
    cb_min = cv2.getTrackbarPos('Cb min', window_name)
    cb_max = cv2.getTrackbarPos('Cb max', window_name)

    lower = np.array([y_min, cr_min, cb_min], dtype=np.uint8)
    upper = np.array([y_max, cr_max, cb_max], dtype=np.uint8)
    return lower, upper


def main():
    image_path = "Testing/YOLO/vid_6/frames/annotated_frame_900.png"  # <- Change this to your image path
    # image_path = "Testing/YOLO/vid_6/frame_50/crop_4.png"  # <- Change this to your image path

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at: {image_path}")
        return

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    window_name = 'YCrCb Thresholding'
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 900, 400)
    create_trackbars(window_name)

    while True:
        lower, upper = get_thresholds(window_name)

        # Create a mask based on the thresholds
        mask = cv2.inRange(img_ycrcb, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Combine original and result side by side
        combined = np.hstack((img, result))
        combined_resized = cv2.resize(combined, (900, 400), interpolation=cv2.INTER_AREA)

        cv2.imshow(window_name, combined_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
