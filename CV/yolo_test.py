from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy

from ultralytics import YOLO
import cv2

# Load the YOLOv11n model
model_ori = YOLO("yolo11n.pt")
model_ori.export(format="engine", device=0)  # Export the model to TensorRT engine format

model = YOLO("yolo11n.engine")  # Load the TensorRT engine model

# Set up the camera source and display output
## Customize the camera pipeline if needed
cam_id = 0
cam_width, cam_height, cam_framerate = 3280, 1848, 28
args = [f"--input-width={cam_width}", f"--input-height={cam_height}", f"--input-rate={cam_framerate}"]
## Use the custom pipeline if needed
camera = videoSource("csi://0", argv=args)
display = videoOutput("display://0")


# Define face class ID (change this based on your model)
FACE_CLASS_ID = 0  # Typically 0 for person, adjust if your model is face-specific

while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        continue
    # Convert the image to a numpy array
    img_np = cudaToNumpy(img)

    # Covert the image to BGR format for OpenCV if needed
    # print("Image shape: ", img_np.shape)
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLO inference
    results = model(img_np, verbose=False, classes=[FACE_CLASS_ID]) # Set verbose=False to suppress output

    # Draw bounding boxes on the image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == FACE_CLASS_ID:
                x1, y1, x2, y2 = box.astype(int)
                label = f"{model.names[class_id]} {conf:.2f}"

                # Draw the bounding box and label on the image
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert the image back to CUDA format for display
    img_display = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_display = cudaFromNumpy(img_display)

    display.Render(img_display)
    display.SetStatus("FPS:{:.1f}".format(display.GetFrameRate()))