from qreader import QReader
import cv2

if __name__ == '__main__':
    # Initialize QReader
    detector = QReader()
    # Read the image
    img = cv2.cvtColor(cv2.imread('qr_in_the_wild.jpg'), cv2.COLOR_BGR2RGB)
    # Detect and decode the QRs within the image
    QRs = detector.detect_and_decode(image=img, return_bboxes=True)
    # Print the results
    for QR in QRs:
        print(QR)
