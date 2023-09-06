from qreader import QReader
import cv2
import os

SAMPLE_IMG = os.path.join(os.path.dirname(__file__), 'documentation', 'resources', 'test_draw_64x64.jpeg')

if __name__ == '__main__':
    # Initialize QReader
    detector = QReader(reencode_to='utf-8')
    # Read the image
    img = cv2.cvtColor(cv2.imread(SAMPLE_IMG), cv2.COLOR_BGR2RGB)
    # Detect and decode the QRs within the image
    QRs = detector.detect_and_decode(image=img, return_detections=True)
    # Print the results
    for QR in QRs:
        print(QR)
