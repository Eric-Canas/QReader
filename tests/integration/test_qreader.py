import os

import cv2
import qrcode
from qrdet import BBOX_XYXY

from qreader import QReader


def test_utf_errors():

    qreader = QReader(model_size="n")
    image_path = "my_image.png"
    data = "â"
    print(f"data = {data}")
    img = qrcode.make(data)

    img.save(image_path)
    img = cv2.imread(image_path)
    os.remove(image_path)
    result = qreader.detect_and_decode(image=img)
    print(f"result = {result[0]}")


def test_decode_test_set():
    images = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "documentation",
            "resources",
            filename,
        )
        for filename in os.listdir(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "documentation", "resources"
            )
        )
    ]
    # Initialize QReader
    detector = QReader(model_size="n")
    # For each image, show the results
    for image_file in images:
        # Read the images
        img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        # Detect and decode the QRs within the image
        decoded_qrs, locations = detector.detect_and_decode(
            image=img, return_detections=True
        )
        # Print the results
        print(f"Image: {image_file} -> {len(decoded_qrs)} QRs detected.")
        for content, location in zip(decoded_qrs, locations):
            print(f"Content: {content}. Position: {tuple(location[BBOX_XYXY])}")
            if content is None:
                pass
                # decoded_qrs = detector.detect_and_decode(image=img, return_detections=False)
        print("-------------------")
