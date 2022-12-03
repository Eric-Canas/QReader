from qreader import QReader
import cv2

if __name__ == '__main__':
    qr = QReader()
    img = cv2.imread('documentation/resources/test_image.jpeg')

    print(qr.detect_and_decode(image=img))