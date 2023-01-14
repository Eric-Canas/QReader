"""
This class implements a robust QR detector & decoder. This detector is based on YOLOv7 model on the side
of the detection and pyzbar on the side of the decoder. It also implements different image preprocessing
techniques. This detector will transparently combine all these techniques to maximize the detection rate
on difficult images.

Mail: eric@ericcanas.com
Date: 02-12-2022
Github: https://github.com/Eric-Canas
"""

from __future__ import annotations

import numpy as np
from pyzbar.pyzbar import decode as decodeQR, ZBarSymbol
import cv2

from qrdet import QRDetector
_SHARPEN_KERNEL = np.array(((-1, -1, -1), (-1, 9, -1), (-1, -1, -1)), dtype=np.float32)

class QReader:
    def __init__(self):
        """
        This class implements a robust, ML Based QR detector & decoder.
        """
        self.detector = QRDetector()

    def detect(self, image: np.ndarray, min_confidence=0.6) -> tuple[tuple[int, int, int, int], ...]:
        """
        This method will detect the QRs in the image and return the bounding boxes of the QR codes. If the QR code is
        not detected, it will return an empty tuple.

        :return: tuple[int, int, int, int], ...]. A tuple with the bounding boxes of the detected QR codes
                                                  in the format (x1, y1, x2, y2). If no QR code is detected, it will
                                                  return an empty tuple.

        """

        detections = self.detector.detect(image=image, return_confidences=True, as_float=False)
        detections = tuple(tuple(detection) for detection, confidence in detections if confidence >= min_confidence)

        return detections

    def decode(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> str | None:
        """
        This method is just a wrapper of pyzbar decodeQR method, that will try to apply image pre-processing when the
        QR code is not readed for the first time.

        :param image: np.ndarray. The image to be read. It must be a np.ndarray (HxWxC) (uint8).
        :param bbox: tuple[int, int, int, int]. The bounding box of the QR code in the format (x1, y1, x2, y2) (can be
                                                obtained with the detect method. If None, it will try to read the QR from
                                                the whole image. Not recommended. Default: None.

        :return: str|None. The decoded content of the QR code or None if it can not be read.
        """
        # Crop the image if a bounding box is given
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]

        # Try to decode the QR code just with pyzbar
        decodedQR = decodeQR(image=image, symbols=[ZBarSymbol.QRCODE])
        if len(decodedQR) == 0:
            # Let's try some desperate postprocessing
            sharpened_img = cv2.cvtColor(cv2.filter2D(src=image, ddepth=-1, kernel=_SHARPEN_KERNEL), cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(sharpened_img, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            decodedQR = decodeQR(image=binary_img, symbols=[ZBarSymbol.QRCODE])
            if len(decodedQR) == 0:
                # Blurring the sharpened image just a bit, works sometimes
                decodedQR = decodeQR(image=cv2.GaussianBlur(src=sharpened_img, ksize=(3, 3), sigmaX=0))

        decodedQR = decodedQR[0].data.decode("utf-8") if len(decodedQR) > 0 else None

        return decodedQR

    def detect_and_decode(self, image: np.ndarray, return_bboxes: bool = False) -> \
            tuple[tuple[tuple[int, int, int, int], str | None], ...] | tuple[str|None, ...]:
        """
        This method will decode all the QR codes in the image and return the decoded results (or None if there is
        a QR but can not be decoded).

        :param image: np.ndarray. np.ndarray (HxWxC). The image to be read. It is expected to be RGB (uint8).
        :param return_bboxes: bool. If True, it will return the bounding boxes of the QR codes. Default: False.

        :return: tuple[tuple[tuple[int, int, int, int], str | None], ...] | tuple[str]. If return_bboxes is True, it will
                    return a tuple with the bounding boxes of the QR codes and the decoded content of the QR codes. If
                    return_bboxes is False, it will return a tuple with the decoded content of the QR codes. The format
                    of the bounding boxes is (x1, y1, x2, y2). The content of the QR codes is a string or None if the QR
                    code can not be decoded.

        """
        bboxes = self.detect(image=image)
        decoded_qrs = tuple(self.decode(image=image, bbox=bbox) for bbox in bboxes)

        if return_bboxes:
            return tuple(zip(bboxes, decoded_qrs))
        else:
            return decoded_qrs
