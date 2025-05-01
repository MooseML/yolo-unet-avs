import os
from tensorflow.keras.models import load_model
from src.detection.yad2k.utils.utils import read_classes, read_anchors

# Point to the real model path based on your project root
BASE_DIR = os.path.join(os.path.dirname(__file__), "../../data/detection/model_data")
BASE_DIR = os.path.abspath(BASE_DIR)

def load_yolo_model(model_path=None):
    """
    Load the pre-trained YOLO model from disk

    Args:
    model_path (str): Optional custom path to the saved YOLO model

    Returns:
    keras.Model: Loaded YOLO model
    """
    model_path = model_path or BASE_DIR
    return load_model(model_path, compile=False)


def load_config(class_path=None, anchor_path=None):
    """
    Load class names and anchors from config files

    Returns:
    tuple: (class_names, anchors)
    """
    class_path = class_path or os.path.join(BASE_DIR, "coco_classes.txt")
    anchor_path = anchor_path or os.path.join(BASE_DIR, "yolo_anchors.txt")

    class_names = read_classes(class_path)
    anchors = read_anchors(anchor_path)

    return class_names, anchors
