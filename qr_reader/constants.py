import os

YOLO_CONFIG_PATH = os.path.join(".", "qr_reader", "yolo_v3", "qrcode-yolov3-tiny.cfg")
YOLO_WEIGHTS_PATH = os.path.join(".", "qr_reader", "yolo_v3", "qrcode-yolov3-tiny_last.weights")

assert os.path.isfile(YOLO_CONFIG_PATH), f"YOLO config file not found at {YOLO_CONFIG_PATH}"
assert os.path.isfile(YOLO_WEIGHTS_PATH), f"YOLO weights file not found at {YOLO_WEIGHTS_PATH}"