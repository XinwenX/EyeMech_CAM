from ultralytics import YOLO

# Load the YOLOv11n model
model_ori = YOLO("yolo11n.pt")
model_ori.export(format="engine", device=0)  # Export the model to TensorRT engine format
