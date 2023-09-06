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
from warnings import warn
import numpy as np
from pyzbar.pyzbar import decode as decodeQR, ZBarSymbol, Decoded
import cv2

from qrdet import QRDetector, BBOX_XYXY

_SHARPEN_KERNEL = np.array(((-1., -1., -1.), (-1., 9., -1.), (-1., -1., -1.)), dtype=np.float32)

class QReader:
    def __init__(self, model_size: str = 's', min_confidence: float = 0.5, reencode_to: str | None = 'shift-jis'):
        """
        This class implements a robust, ML Based QR detector & decoder.

        :param model_size: str. The size of the model to use. It can be 'n' (nano), 's' (small), 'm' (medium) or
                                'l' (large). Larger models are more accurate but slower. Default (and recommended): 's'.
        :param min_confidence: float. The minimum confidence of the QR detection to be considered valid. Default: 0.5.
        :param reencode_to: str or None. The encoding to reencode the decoded QR code. If None, it will do just a one-step
        encoding to utf-8. If you find some characters being decoded incorrectly, try to reencode to a
        [Code Page](https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers) that matches your
        specific charset. Recommendations that have been found useful:
            - 'shift-jis' for Germanic languages
            - 'cp65001' for Asian languages (Thanks to @nguyen-viet-hung for the suggestion)
        """
        self.detector = QRDetector(conf_th=min_confidence)
        self.reencode_to = reencode_to

    def detect(self, image: np.ndarray) -> tuple[dict[str, np.ndarray|float|tuple[float|int, float|int]]]:
        """
        This method will detect the QRs in the image and return a tuple of dictionaries with all the detection
        information.

        :return: tuple[dict[str, np.ndarray|float|tuple[float|int, float|int]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quadrilateral_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2).
                                              Fitted from the polygon.
            - 'expanded_quadrilateral_xy': np.ndarray. An expanded version of quadrilateral_xy, with shape (4, 2),
                that always include all the points within polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
        """
        return self.detector.detect(image=image)

    def decode(self, image: np.ndarray, detection_result: dict[str, np.ndarray|float|tuple[float|int, float|int]]) -> \
            str | None:
        """
        This method is a wrapper of pyzbar decodeQR method, that takes profit of the information of each
         detection_result to apply image pre-processing techniques that will maximize the decoding rate.

        :param image: np.ndarray. The image to be read. It must be a np.ndarray (HxWxC) (uint8).
        :param detection_result: dict[str, np.ndarray|float|tuple[float|int, float|int]]. One of the detection dicts
            returned by the detect method. Note that QReader.detect() returns a tuple of these dicts. This method
            expects just one of them.
        :return: str|None. The decoded content of the QR code or None if it can not be read.
        """
        # Crop the image if a bounding box is given
        bbox = detection_result[BBOX_XYXY]
        x1, y1, x2, y2 = tuple(int(coord) for coord in bbox)
        image = image[y1:y2, x1:x2]

        for resize_factor in (1, 0.5, 2, 0.33, 3, 0.25):
            # Don't exceed 1024 image size limit (Nothing will be better decoded beyond this size)
            if resize_factor > 1 and (image.shape[0] * resize_factor > 1024 or image.shape[1] * resize_factor > 1024):
                continue
            resized_image = cv2.resize(image, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
            # Try to decode the QR code just with pyzbar
            decodedQR = self.decode_qr_zbar(image=resized_image)
            if len(decodedQR) > 0:
                decoded_str = decodedQR[0].data.decode('utf-8')
                if self.reencode_to is not None and self.reencode_to != 'utf-8':
                    try:
                        decoded_str = decoded_str.encode(self.reencode_to).decode('utf-8')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        # When double decoding fails, just return the first decoded string with utf-8
                        warn(f'Double decoding failed for {self.reencode_to}. Returning utf-8 decoded string.')
                return decoded_str
        return None

    def decode_qr_zbar(self, image: np.ndarray)-> list[Decoded]:
        """
        Try to decode the QR code just with pyzbar, pre-processing the image if it fails in different ways that
        sometimes work.
        :param image: np.ndarray. The image to be read. It must be a np.ndarray (HxWxC) (uint8).
        :return: tuple. The decoded QR code in the zbar format.
        """
        # Try to just decode the QR code
        decodedQR = decodeQR(image=image, symbols=[ZBarSymbol.QRCODE])
        if len(decodedQR) > 0:
            return decodedQR

        # If it not works, try to parse to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        decodedQR = self.__threshold_and_blur_decodings(image=gray, blur_kernel_sizes=((5, 5), (7, 7)))
        if len(decodedQR) > 0:
            return decodedQR

        # If it not works, try to sharpen the image
        sharpened_gray = cv2.cvtColor(cv2.filter2D(src=image, ddepth=-1, kernel=_SHARPEN_KERNEL),
                                     cv2.COLOR_RGB2GRAY)
        decodedQR = self.__threshold_and_blur_decodings(image=sharpened_gray, blur_kernel_sizes=((3, 3),))
        if len(decodedQR) > 0:
            return decodedQR

        return []

    def __threshold_and_blur_decodings(self, image: np.ndarray, blur_kernel_sizes: tuple[tuple[int, int]] = ((3, 3), )) ->\
            list[Decoded]:
        """
        Try to decode the QR code just with pyzbar, pre-processing the image with different blur and threshold
        filters.
        :param image: np.ndarray. The image to be read. It must be a 2D or 3D np.ndarray (HxW[xC]) (uint8).
        :return: list[Decoded]. The decoded QR code/s in the zbar format. If it fails, it will return an empty list.
        """

        assert 2 <= len(image.shape) <= 3, f"image must be 2D or 3D (HxW[xC]) (uint8). Got {image.shape}"
        decodedQR = decodeQR(image=image, symbols=[ZBarSymbol.QRCODE])
        if len(decodedQR) > 0:
            return decodedQR

        # Try to binarize the image (Only works with 2D images)
        if len(image.shape) == 2:
            _, binary_image = cv2.threshold(image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            decodedQR = decodeQR(image=binary_image, symbols=[ZBarSymbol.QRCODE])
            if len(decodedQR) > 0:
                return decodedQR

        for kernel_size in blur_kernel_sizes:
            assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, \
                f"kernel_size must be a tuple of 2 elements. Got {kernel_size}"
            assert all(kernel_size[i] % 2 == 1 for i in range(2)), \
                f"kernel_size must be a tuple of odd elements. Got {kernel_size}"

            # If it not works, try to parse to sharpened grayscale
            blur_image = cv2.GaussianBlur(src=image, ksize=kernel_size, sigmaX=0)
            decodedQR = decodeQR(image=blur_image, symbols=[ZBarSymbol.QRCODE])
            if len(decodedQR) > 0:
                return decodedQR
        return []


    def detect_and_decode(self, image: np.ndarray, return_detections: bool = False) -> \
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
        detections = self.detect(image=image)
        decoded_qrs = tuple(self.decode(image=image, detection_result=detection) for detection in detections)


        if return_detections:
            return tuple(zip(detections, decoded_qrs))
        else:
            return decoded_qrs
