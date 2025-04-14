import cv2
import numpy as np

def nothing(x):
    pass

def create_trackbars(window_name):
    # Create trackbars for lower and upper threshold values for H, S, V
    cv2.createTrackbar('H min', window_name, 0, 179, nothing)
    cv2.createTrackbar('H max', window_name, 179, 179, nothing)
    cv2.createTrackbar('S min', window_name, 0, 255, nothing)
    cv2.createTrackbar('S max', window_name, 255, 255, nothing)
    cv2.createTrackbar('V min', window_name, 0, 255, nothing)
    cv2.createTrackbar('V max', window_name, 255, 255, nothing)

def get_thresholds(window_name):
    h_min = cv2.getTrackbarPos('H min', window_name)
    h_max = cv2.getTrackbarPos('H max', window_name)
    s_min = cv2.getTrackbarPos('S min', window_name)
    s_max = cv2.getTrackbarPos('S max', window_name)
    v_min = cv2.getTrackbarPos('V min', window_name)
    v_max = cv2.getTrackbarPos('V max', window_name)

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lower, upper

def main():
    image_path = "../Testing/YOLO/vid_6/frames/annotated_frame_900.png"  # <- Change this to your image path
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at: {image_path}")
        return

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    window_name = 'HSV Thresholding'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 400)  # Resize the main window

    create_trackbars(window_name)

    while True:
        lower, upper = get_thresholds(window_name)

        # Create a mask based on the thresholds
        mask = cv2.inRange(img_hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Resize for display
        combined = np.hstack((img, result))
        combined_resized = cv2.resize(combined, (900, 400), interpolation=cv2.INTER_AREA)

        cv2.imshow(window_name, combined_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
