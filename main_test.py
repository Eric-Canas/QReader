from qreader import QReader
import cv2
from pyzbar.pyzbar import decode

if __name__ == '__main__':
    # Initialize the three tested readers (QRReader, OpenCV and pyzbar)
    qreader_reader, cv2_reader, pyzbar_reader = QReader(), cv2.QRCodeDetector(), decode

    for img_path in ('test_mobile.jpeg', 'test_draw_64x64.jpeg'):
        # Read the image
        img = cv2.imread(img_path)

        # Try to decode the QR code with the three readers
        qreader_out = qreader_reader.detect_and_decode(image=img)
        cv2_out = cv2_reader.detectAndDecode(img=img)[0]
        pyzbar_out = pyzbar_reader(image=img)
        # Read the content of the pyzbar output
        pyzbar_out = pyzbar_out[0].data.decode('utf-8') if len(pyzbar_out) > 0 else ""

        # Print the results
        print(f"Image: {img_path} -> QReader: {qreader_out}. OpenCV: {cv2_out}. pyzbar: {pyzbar_out}.")