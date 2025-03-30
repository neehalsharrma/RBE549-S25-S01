from ultralytics import YOLO

# Load model
def load_model():
    model = YOLO("yolo11l.yaml").load("yolo11l.pt")  # build from YAML and transfer weights
    return model