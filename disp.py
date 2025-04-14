import numpy as np
import cv2
import matplotlib.pyplot as plt


# 경로 설정
depth_jpeg_path = "frame_00005_disp.jpeg"
disparity_npy_path = "frame_00005_disp.npy"

# Load image and disparity
depth_img = cv2.imread(depth_jpeg_path)
depth_img_rgb = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)

disparity_map = np.load(disparity_npy_path)
disparity_map = np.squeeze(disparity_map)  # (1, 1, H, W) → (H, W)

# Resize image to match disparity shape
resized_img = cv2.resize(depth_img, (disparity_map.shape[1], disparity_map.shape[0]))

# KITTI camera parameters
f_kitti = 721.5377
B_kitti = 0.5327

# RealSense D415
f_realsense = 640
B_realsense = 0.05

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity_value = disparity_map[y, x]
        pixel_bgr = resized_img[y, x]
        brightness = int(0.299 * pixel_bgr[2] + 0.587 * pixel_bgr[1] + 0.114 * pixel_bgr[0])

        if disparity_value > 0:
            depth_kitti = (f_kitti * B_kitti) / disparity_value
            depth_realsense = (f_realsense * B_realsense) / disparity_value
            print(f"(x={x}, y={y}) → Disparity: {disparity_value:.2f}, Brightness: {brightness}, Depth (KITTI): {depth_kitti:.3f} m, Depth (RealSense): {depth_realsense:.3f} m")

            # print(f"(x={x}, y={y}) → Disparity: {disparity_value:.2f}, Brightness: {brightness}, scaled_depth: {depth_iphone:.3f} m")
        else:
            print(f"(x={x}, y={y}) → Disparity: {disparity_value:.2f}, Brightness: {brightness}, Depth: undefined (disparity = 0)")

cv2.namedWindow("Depth (click to inspect)")
cv2.setMouseCallback("Depth (click to inspect)", click_event)
cv2.imshow("Depth (click to inspect)", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()