from qrdet import BBOX_XYXY

from qreader import QReader
import cv2
import os

SAMPLE_IMG = os.path.join(os.path.dirname(__file__), 'documentation', 'resources', 'test_draw_64x64.jpeg')

images = [os.path.join(os.path.dirname(__file__), 'testset', filename)
          for filename in os.listdir(os.path.join(os.path.dirname(__file__), 'testset'))]

if __name__ == '__main__':
    # Initialize QReader
    detector = QReader(model_size='n')
    # For each image, show the results
    for image_file in images:
        # Read the images
        img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        # Detect and decode the QRs within the image
        decoded_qrs, locations = detector.detect_and_decode(image=img, return_detections=True)
        # Print the results
        print(f"Image: {image_file} -> {len(decoded_qrs)} QRs detected.")
        for content, location in zip(decoded_qrs, locations):
            print(f"Content: {content}. Position: {tuple(location[BBOX_XYXY])}")
            if content is None:
                pass
                #decoded_qrs = detector.detect_and_decode(image=img, return_detections=False)
        print('-------------------')
