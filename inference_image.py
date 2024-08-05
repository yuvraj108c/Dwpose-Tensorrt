import cv2
import os
from dwpose import DWposeDetector

frame = cv2.imread("./assets/usain-bolt.jpg")

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

dwpose = DWposeDetector()
output = dwpose(image_np_hwc=frame, show_body=True,
                show_face=True, show_hands=True)

cv2.imwrite(os.path.join(out_dir, "result.jpg"), output)
