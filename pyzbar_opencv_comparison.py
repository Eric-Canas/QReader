import os

from qreader import QReader
import cv2
from pyzbar.pyzbar import decode

SAMPLE_IMG_1 = os.path.join(os.path.dirname(__file__), 'documentation', 'resources', 'test_draw_64x64.jpeg')
SAMPLE_IMG_2 = os.path.join(os.path.dirname(__file__), 'documentation', 'resources', '64x64.png')

if __name__ == '__main__':
    # Initialize the three tested readers (QRReader, OpenCV and pyzbar)
    qreader_reader, cv2_reader, pyzbar_reader = QReader(), cv2.QRCodeDetector(), decode


    for img_path in (SAMPLE_IMG_1, SAMPLE_IMG_2):
        # Read the image
        img = cv2.imread(img_path)

        # Try to decode the QR code with the three readers
        qreader_out = qreader_reader.detect_and_decode(image=img)
        cv2_out = cv2_reader.detectAndDecode(img=img)[0]
        pyzbar_out = pyzbar_reader(image=img)
        # Read the content of the pyzbar output
        pyzbar_out = tuple(out.data.decode('utf-8') for out in pyzbar_out)

        # Print the results
        print(f"Image: {img_path} -> QReader: {qreader_out}. OpenCV: {cv2_out}. pyzbar: {pyzbar_out}.")