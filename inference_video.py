import cv2
import os
from dwpose import DWposeDetector
import timeit

video_file = "./assets/skate.mp4"

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

dwpose = DWposeDetector()
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(out_dir, "result.mp4"),
                      fourcc, fps, (frame_width, frame_height))

start = timeit.default_timer()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output = dwpose(image_np_hwc=frame, show_body=True,
                    show_face=True, show_hands=True)
    out.write(output)

cap.release()
out.release()

end = timeit.default_timer()
time_taken = end-start

print(f"Video processing completed in {time_taken} seconds, FPS: {frame_count/time_taken}")
