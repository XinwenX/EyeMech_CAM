from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaDrawLine

from ultralytics import YOLO
import torch

# Load the YOLOv11n Tensor model
model = YOLO("yolo11n.engine")  # Load the TensorRT engine model

# Set up the camera source and display output
## Customize the camera pipeline if needed
cam_id = 0
cam_width, cam_height, cam_framerate = 3280, 1848, 28
args = [
    f"--input-width={cam_width}", 
    f"--input-height={cam_height}", 
    f"--input-rate={cam_framerate}"
]
## Use the custom pipeline if needed
camera = videoSource("csi://0", argv=args)
display = videoOutput("display://0")

FACE_CLASS_ID = 0
while display.IsStreaming():
    cuda_img = camera.Capture()
    if cuda_img is None:
        continue

    results = model.predict(
        cudaToNumpy(cuda_img),  # Convert CUDA image to numpy for inference
        device='cuda',  # Use CUDA for inference
        verbose=False,  # Suppress output
        classes=[FACE_CLASS_ID],  # Filter for specific class (e.g., face)
    )
    
    for result in results:
        if result.boxes is not None:
            # Draw bounding boxes on the image
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
            for x1, y1, x2, y2 in boxes:
                cudaDrawLine(cuda_img, (x1, y1), (x2, y1), (0, 255, 0, 255))  # Top
                cudaDrawLine(cuda_img, (x2, y1), (x2, y2), (0, 255, 0, 255))  # Right
                cudaDrawLine(cuda_img, (x2, y2), (x1, y2), (0, 255, 0, 255))  # Bottom
                cudaDrawLine(cuda_img, (x1, y2), (x1, y1), (0, 255, 0, 255))  # Left    
    display.Render(cuda_img)
    display.SetStatus(f"FPS: {display.GetFrameRate():.1f}")