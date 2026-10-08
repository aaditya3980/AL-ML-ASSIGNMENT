import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLO model (lightweight 'nano' version for fast execution)
model = YOLO('yolov8n.pt')

# 2. Open CCTV footage / video file (replace 'cctv_footage.mp4' with your file name)
# Note: Use cv2.VideoCapture(0) to use your webcam if you don't have a video file.
video_path = 'cctv_footage.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file or CCTV stream.")
    exit()

print("Processing CCTV Footage. Press 'q' on the video window to exit...")

# 3. Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # 4. Perform object detection on the current frame
    results = model(frame)

    # 5. Visualize and get annotated frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # 6. Display the output frame in a GUI window
    cv2.imshow("CCTV Object Detection (YOLO)", annotated_frame)

    # Exit stream if user presses 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Release resources
cap.release()
cv2.destroyAllWindows()
