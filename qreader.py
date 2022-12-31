"""
This class implements a robust QR detector & decoder. This detector is based on several other detectors and decoders,
such as pyzbar, openCV and YoloV3, as well as different image preprocessing techniques. This detector will transparently
combine all these techniques to maximize the detection rate on difficult images (e.g. QR code too small).

Mail: eric@ericcanas.com
Date: 02-12-2022
Github: https://github.com/Eric-Canas
"""

from __future__ import annotations

import numpy as np
from pyzbar.pyzbar import decode as decodeQR, ZBarSymbol
from cv2 import QRCodeDetector, resize
import cv2

from __yolo_v3_qr_detector.yolov3_qr_detector import _YoloV3QRDetector

_SHARPEN_KERNEL = np.array(((-1, -1, -1), (-1, 9, -1), (-1, -1, -1)), dtype=np.float32)

class QReader:
    def __init__(self):
        """
        This class implements a robust QR detector & decoder.
        """
        self.__cv2_detector = QRCodeDetector()
        self.__yolo_v3_detector = _YoloV3QRDetector()

    def detect(self, image: np.ndarray) -> tuple[bool, tuple[int, int, int, int]|None]:
        """
        This method will detect the QR code in the image and return the bounding box of the QR code. If the QR code is
        not detected, it will return None.

        This method will always assume that there is only one QR code in the image.

        :param image: np.ndarray. The image to be read. It must be an uint8 np.ndarray (HxWxC).

        :return: tuple[bool, tuple[int, int, int, int]|None]. The first element is a boolean indicating if the QR code
                                                              was detected or not. The second element is the bounding
                                                              box of the QR code in the format (x1, y1, x2, y2) or None
                                                              if it was not detected.
        """
        found, bbox = self.__yolo_v3_detector.detect(img=image)
        if not found:
            try:
                found, bbox = self.__cv2_detector.detect(img=image)
                if found:
                    h, w = image.shape[:2]
                    x1, y1 = max(round(np.min(bbox[0, :, 0])), 0), max(round(np.min(bbox[0, :, 1])), 0)
                    x2, y2 = min(round(np.max(bbox[0, :, 0])), w), min(round(np.max(bbox[0, :, 1])), h)
                    bbox = (x1, y1, x2, y2)
            except cv2.error:
                found, bbox = False, None

        return found, bbox

    def decode(self, image: np.ndarray, bbox: tuple[int, int, int, int]|None = None) -> str | None:
        """
        This method is just a wrapper of pyzbar decodeQR method, that will try to apply image pre-processing when the
        QR code is not detected for the first time.

        :param image: np.ndarray. The image to be read. It must be a np.ndarray (HxWxC) (uint8).
        :param bbox: tuple[int, int, int, int]. The bounding box of the QR code in the format (x1, y1, x2, y2) or None.
                                                If not given it will try to find it in the whole image.

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

    def detect_and_decode(self, image: np.ndarray, deep_search: bool = True) -> str | None:
        """
        This method will decode the QR code in the image and return the result. If the QR code is not detected, it will
        return None.

        :param image: np.ndarray. The image to be read. It can be a path or a np.ndarray (HxWxC)
        :param deep_search: bool. If True, it will make a deep search if the QR can't be detected in the first attempt.
                                  This deep search will inspect subregions of the image to locate difficult QR codes.
                                  It can be slightly slower but severally increases the detection rate. Default: True.

        :return: str|None. The result of the detection or None if it is not detected
        """
        if not deep_search:
            found, bbox = self.detect(image=image)
            if not found:
                return None
            return self.decode(image=image, bbox=bbox)
        else:
            return self.__deep_detect_and_decode(image=image)

    def __deep_detect_and_decode(self, image: np.ndarray, min_img_size = 64) -> str | None:
        """
        This method will try to detect and decode the QR code in the image. If it is not detected, it will
        fall back to a deep search that will inspect subregions of the image, looking for difficult or small QR codes.

        :param image: np.ndarray. The image to be read. It must be an uint8 numpy array (HxW[xC]).
        :param min_img_size: int. The minimum size of the image to be inspected. No crops will be done below this size.
                                  Default: 64.

        :return: str|None. The decoded QR content or None if it couldn't be detected.
        """
        img_h, img_w = image.shape[:2]
        current_h, current_w = img_h, img_w

        # Starting with the whole image, try sub-patches of the image until a QR code is detected
        while current_h >= min_img_size and current_w >= min_img_size:
            # Include some overlap between sub-patches to avoid missing QR codes
            h_pad, w_pad = current_h // 4, current_w // 4

            for x in range(0, img_w, current_w):
                for y in range(0, img_h, current_h):
                    subimage = image[max(0, y - h_pad):min(img_h, y + current_h + h_pad),
                                     max(0, x - w_pad):min(img_w, x + current_w + w_pad)]
                    # Try to decode it applying resizing as fallback
                    decodedQR = self.__deep_detect_and_decode_with_resize(image=subimage)
                    if decodedQR is not None:
                        return decodedQR

            current_h, current_w = current_h // 2, current_w // 2

        return None

    def __deep_detect_and_decode_with_resize(self, image: np.ndarray,
                                             rescale_factors: tuple[float|int, ...] = (1, 2, 0.5, 4, 0.25),
                                             min_img_size_to_check: int = 16, max_image_size_to_check: int = 2048) -> str | None:
        """
        This method will try to find QR codes that are difficult to read because they are too small or too big.
        The detection step will run in the whole image, then, if something is found, the decoding step will iteratively
        run in rescaled versions of the image aiming to decode the QR code at one of these sizes.

        :param image: np.ndarray. The image to be read. It must be an uint8 numpy array (HxW[xC]).
        :param rescale_factors: tuple[float|int, ...]. Factors to be used for rescaling the image if the QR is detected
                                                       Default: (1, 2, 0.5, 4, 0.25).
        :param min_img_size_to_check: int. The minimum size of the image to be inspected. No resizes will be
                                           done below this size. Default: 16.
        :param max_image_size_to_check: int. The maximum size of the image to be inspected. No resizes will be
                                             done above this size. Default: 2048.

        :return: str|None. The decodedQR or None if it couldn't be detected
        """

        # Try to detect the QR code in the whole image
        found, bbox = self.detect(image)

        if found:
            # Crop the QR code
            x1, y1, x2, y2 = bbox
            cropped_img = image[y1:y2, x1:x2]

            # Try to decode the QR code by iteratively rescaling the image
            for rescale in rescale_factors:
                # Avoid to rescale the image too big or too small
                if any(not min_img_size_to_check < axis*rescale < max_image_size_to_check for axis in cropped_img.shape[:2]):
                    continue

                resized_img = resize(src=cropped_img, dsize=None, fx=rescale, fy=rescale, interpolation=cv2.INTER_CUBIC)

                decodedQR = self.decode(image=resized_img, bbox=None)
                # If something is decoded, return it
                if decodedQR is not None:
                    return decodedQR

        return None
