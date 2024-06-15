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

import os
import typing
from dataclasses import dataclass
from warnings import warn

import cv2
import numpy as np
from pyzbar.pyzbar import Decoded, ZBarSymbol
from pyzbar.pyzbar import decode as decodeQR
from qrdet import (
    BBOX_XYXY,
    CONFIDENCE,
    CXCY,
    PADDED_QUAD_XY,
    POLYGON_XY,
    QUAD_XY,
    WH,
    QRDetector,
    crop_qr,
)

_SHARPEN_KERNEL = np.array(
    ((-1.0, -1.0, -1.0), (-1.0, 9.0, -1.0), (-1.0, -1.0, -1.0)), dtype=np.float32
)

# In windows shift-jis is the default encoding will use, while in linux is big5
DEFAULT_REENCODINGS = (
    ("shift-jis", "big5") if os.name == "nt" else ("big5", "shift-jis")
)


@dataclass(frozen=True)
class DecodeQRResult:
    scale_factor: float
    corrections: typing.Literal["cropped_bbox", "corrected_perspective"]
    flavor: typing.Literal["original", "inverted", "grayscale"]
    blur_kernel_sizes: tuple[tuple[int, int]] | None
    image: np.ndarray
    result: Decoded


def wrap(
    scale_factor: float,
    corrections: str,
    flavor: str,
    blur_kernel_sizes: tuple[tuple[int, int]],
    image: np.ndarray,
    results: typing.List[Decoded],
) -> list[DecodeQRResult]:

    return [
        DecodeQRResult(
            scale_factor=scale_factor,
            corrections=corrections,
            flavor=flavor,
            blur_kernel_sizes=blur_kernel_sizes,
            image=image,
            result=result,
        )
        for result in results
    ]


class QReader:
    def __init__(
        self,
        model_size: str = "s",
        min_confidence: float = 0.5,
        reencode_to: str | tuple[str] | list[str] | None = DEFAULT_REENCODINGS,
        weights_folder: str | None = None,
    ):
        """
        This class implements a robust, ML Based QR detector & decoder.

        :param model_size: str. The size of the model to use. It can be 'n' (nano), 's' (small), 'm' (medium) or
                                'l' (large). Larger models are more accurate but slower. Default (and recommended): 's'.
        :param  min_confidence: float. The minimum confidence of the QR detection to be considered valid. Values closer to 0.0 can get more
                False Positives, while values closer to 1.0 can lose difficult QRs. Default (and recommended): 0.5.
        :param reencode_to: str or None. The encoding to reencode the decoded QR code. If None, it will do just a one-step
        encoding to utf-8. If you find some characters being decoded incorrectly, try to reencode to a
        [Code Page](https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers) that matches your
        specific charset. Recommendations that have been found useful:
            - 'shift-jis' for Germanic languages
            - 'cp65001' for Asian languages (Thanks to @nguyen-viet-hung for the suggestion)

        :param weights_folder: str or None. The folder where the weights of the model will be stored. If None, they will
        be stored in the default folder of the qrdet package.
        """
        if weights_folder is None:
            self.detector = QRDetector(model_size=model_size, conf_th=min_confidence)
        else:
            self.detector = QRDetector(
                model_size=model_size,
                conf_th=min_confidence,
                weights_folder=weights_folder,
            )

        if isinstance(reencode_to, str):
            self.reencode_to = (reencode_to,) if reencode_to != "utf-8" else ()
        elif reencode_to is None:
            self.reencode_to = ()
        else:
            assert isinstance(
                reencode_to, (tuple, list)
            ), f"reencode_to must be a str, tuple, list or None. Got {type(reencode_to)}"
            self.reencode_to = reencode_to

    def detect(
        self, image: np.ndarray, is_bgr: bool = False
    ) -> tuple[dict[str, np.ndarray | float | tuple[float | int, float | int]]]:
        """
        This method will detect the QRs in the image and return a tuple of dictionaries with all the detection
        information.

        :param image: np.ndarray. The image to be read. It is expected to be RGB or BGR (uint8). Format (HxWx3).
        :param is_bgr: bool.  If True, the received image is expected to be BGR instead of RGB.

        :return: tuple[dict[str, np.ndarray|float|tuple[float|int, float|int]]]. A tuple of dictionaries containing the
            following keys:
            - 'confidence': float. The confidence of the detection.
            - 'bbox_xyxy': np.ndarray. The bounding box of the detection in the format [x1, y1, x2, y2].
            - 'cxcy': tuple[float, float]. The center of the bounding box in the format (x, y).
            - 'wh': tuple[float, float]. The width and height of the bounding box in the format (w, h).
            - 'polygon_xy': np.ndarray. The accurate polygon that surrounds the QR code, with shape (N, 2).
            - 'quad_xy': np.ndarray. The quadrilateral that surrounds the QR code, with shape (4, 2).
                                              Fitted from the polygon.
            - 'padded_quad_xy': np.ndarray. An expanded version of quad_xy, with shape (4, 2), that always include
                                all the points within polygon_xy.

            All these keys (except 'confidence') have a 'n' (normalized) version. For example, 'bbox_xyxy' is the
            bounding box in absolute coordinates, while 'bbox_xyxyn' is the bounding box in normalized coordinates
            (from 0. to 1.).
        """
        return self.detector.detect(image=image, is_bgr=is_bgr)

    def decode(
        self,
        image: np.ndarray,
        detection_result: dict[
            str, np.ndarray | float | tuple[float | int, float | int]
        ],
    ) -> str | None:
        """
        This method decodes a single QR code on the given image, described by a detection_result.

        Internally, this method will run the pyzbar decoder, using the information of the detection_result, to apply
        different image preprocessing techniques that heavily increase the decoding rate.

        :param image: np.ndarray. NumPy Array with the ``image`` that contains the _QR_ to decode. The image is
                                expected to be in ``uint8`` format [_HxWxC_], RGB.
        :param detection_result: dict[str, np.ndarray|float|tuple[float|int, float|int]]. One of the detection dicts
            returned by the detect method. Note that QReader.detect() returns a tuple of these dicts. This method
            expects just one of them.
        :return: str|None. The decoded content of the QR code or None if it can not be read.
        """
        # Crop the image if a bounding box is given
        decodedQR = self._decode_qr_zbar(image=image, detection_result=detection_result)
        if len(decodedQR) > 0:
            # Take first result only
            decodeQRResult = decodedQR[0]
            decoded_str = decodeQRResult.result.data.decode("utf-8")
            for encoding in self.reencode_to:
                try:
                    decoded_str = decoded_str.encode(encoding).decode("utf-8")
                    break
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
            else:
                if len(self.reencode_to) > 0:
                    # When double decoding fails, just return the first decoded string with utf-8
                    warn(
                        f"Double decoding failed for {self.reencode_to}. Returning utf-8 decoded string."
                    )

            return decoded_str
        return None

    def detect_and_decode(
        self, image: np.ndarray, return_detections: bool = False, is_bgr: bool = False
    ) -> (
        tuple[
            dict[str, np.ndarray | float | tuple[float | int, float | int]], str | None
        ]
        | tuple[str | None, ...]
    ):
        """
        This method will decode the **QR** codes in the given image and return the decoded strings
        (or None, if any of them was detected but not decoded).

        :param image: np.ndarray. np.ndarray (HxWx3). The image to be read. It is expected to be RGB or BGR (uint8).
        :param return_detections: bool. If True, it will return the full detection results together with the decoded
                                        QRs. If False, it will return only the decoded content of the QR codes.
        :param is_bgr: bool. If True, the received image is expected to be BGR instead of RGB.

        :return: tuple[dict[str, np.ndarray | float | tuple[float | int, float | int]], str | None] | tuple[str | None, ...]
                    If return_detections is True, it will return a tuple of tuples. Each tuple will contain the
                    detection result (a dictionary with the keys 'confidence', 'bbox_xyxy', 'polygon_xy'...) and the
                    decoded QR code (or None if it can not be decoded). If return_detections is False, it will return
                    a tuple of strings with the decoded QR codes (or None if it can not be decoded).

        """
        detections = self.detect(image=image, is_bgr=is_bgr)
        decoded_qrs = tuple(
            self.decode(image=image, detection_result=detection)
            for detection in detections
        )

        if return_detections:
            return decoded_qrs, detections
        else:
            return decoded_qrs

    def get_detection_result_from_polygon(
        self,
        quadrilateral_xy: (
            np.ndarray
            | tuple[tuple[float | int, float | int], ...]
            | list[list[float | int, float | int]]
        ),
    ) -> dict[str, np.ndarray | float | tuple[float | int, float | int]]:
        """
        This method will simulate a detection result from the given quadrilateral. This is useful when you have detected
        a QR code with a different detector and you want to use this class to decode it.
        :param quadrilateral_xy: np.ndarray|tuple|list. The quadrilateral that surrounds the QR code, with shape (4, 2).
                                In list, tuple or np.ndarray format. Example: ((x1, y1), (x2, y2), (x3, y3), (x4, y4)).
                                It must be in absolute coordinates (not normalized).
        :return: dict[str, np.ndarray|float|tuple[float|int, float|int]]. A dictionary that is compatible with
        the detection_result parameter of the decode method.
        """
        assert isinstance(
            quadrilateral_xy, (np.ndarray, tuple, list)
        ), f"quadrilateral_xy must be a np.ndarray, tuple or list. Got {type(quadrilateral_xy)}"
        assert (
            len(quadrilateral_xy) == 4
        ), f"quadrilateral_xy must have 4 points. Got {len(quadrilateral_xy)}"
        assert all(
            len(point) == 2 for point in quadrilateral_xy
        ), f"Each point in quadrilateral_xy must have 2 coordinates (X, Y). Got {quadrilateral_xy}"

        polygon = np.array(quadrilateral_xy, dtype=np.float32)
        bbox_xyxy = np.array(
            [
                polygon[:, 0].min(),
                polygon[:, 1].min(),
                polygon[:, 0].max(),
                polygon[:, 1].max(),
            ],
            dtype=np.float32,
        )
        cxcy = ((bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2)
        wh = (bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1])
        return {
            CONFIDENCE: 1.0,
            BBOX_XYXY: bbox_xyxy,
            CXCY: cxcy,
            WH: wh,
            POLYGON_XY: polygon,
            QUAD_XY: polygon.copy(),
            PADDED_QUAD_XY: polygon.copy(),
        }

    def _decode_qr_zbar(
        self,
        image: np.ndarray,
        detection_result: dict[
            str, np.ndarray | float | tuple[float | int, float | int]
        ],
    ) -> list[DecodeQRResult]:
        """
        Try to decode the QR code just with pyzbar, pre-processing the image if it fails in different ways that
        sometimes work.
        :param image: np.ndarray. The image to be read. It must be a np.ndarray (HxWxC) (uint8).
        :param detection_result: dict[str, np.ndarray|float|tuple[float|int, float|int]]. One of the detection dicts
            returned by the detect method. Note that QReader.detect() returns a tuple of these dicts. This method
            expects just one of them.
        :return: tuple. The decoded QR code in the zbar format.
        """
        # Crop the QR for bbox and quad
        cropped_bbox, _ = crop_qr(
            image=image, detection=detection_result, crop_key=BBOX_XYXY
        )
        cropped_quad, updated_detection = crop_qr(
            image=image, detection=detection_result, crop_key=PADDED_QUAD_XY
        )
        corrected_perspective = self.__correct_perspective(
            image=cropped_quad, padded_quad_xy=updated_detection[PADDED_QUAD_XY]
        )

        for scale_factor in (1, 0.5, 2, 0.25, 3, 4):
            for label, image in {
                "cropped_bbox": cropped_bbox,
                "corrected_perspective": corrected_perspective,
            }.items():
                # If rescaled_image will be larger than 1024px, skip it
                # TODO: Decide a minimum size for the QRs based on the resize benchmark
                if (
                    not all(25 < axis < 1024 for axis in image.shape[:2])
                    and scale_factor != 1
                ):
                    continue

                rescaled_image = cv2.resize(
                    src=image,
                    dsize=None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_CUBIC,
                )
                decodedQR = decodeQR(image=rescaled_image, symbols=[ZBarSymbol.QRCODE])
                if len(decodedQR) > 0:
                    return wrap(
                        scale_factor=scale_factor,
                        corrections=label,
                        flavor="original",
                        blur_kernel_sizes=None,
                        image=rescaled_image,
                        results=decodedQR,
                    )
                # For QRs with black background and white foreground, try to invert the image
                inverted_image = image = 255 - rescaled_image
                decodedQR = decodeQR(inverted_image, symbols=[ZBarSymbol.QRCODE])
                if len(decodedQR) > 0:
                    return wrap(
                        scale_factor=scale_factor,
                        corrections=label,
                        flavor="inverted",
                        blur_kernel_sizes=None,
                        image=inverted_image,
                        results=decodedQR,
                    )

                # If it not works, try to parse to grayscale (if it is not already)
                if len(rescaled_image.shape) == 3:
                    assert (
                        rescaled_image.shape[2] == 3
                    ), f"Image must be RGB or BGR, but it has {image.shape[2]} channels."
                    gray = cv2.cvtColor(rescaled_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = rescaled_image
                decodedQR = self.__threshold_and_blur_decodings(
                    image=gray, blur_kernel_sizes=((5, 5), (7, 7))
                )
                if len(decodedQR) > 0:
                    return wrap(
                        scale_factor=scale_factor,
                        corrections=label,
                        flavor="grayscale",
                        blur_kernel_sizes=((5, 5), (7, 7)),
                        image=gray,
                        results=decodedQR,
                    )

                if len(rescaled_image.shape) == 3:
                    # If it not works, try to sharpen the image
                    sharpened_gray = cv2.cvtColor(
                        cv2.filter2D(
                            src=rescaled_image, ddepth=-1, kernel=_SHARPEN_KERNEL
                        ),
                        cv2.COLOR_RGB2GRAY,
                    )
                else:
                    sharpened_gray = cv2.filter2D(
                        src=rescaled_image, ddepth=-1, kernel=_SHARPEN_KERNEL
                    )
                decodedQR = self.__threshold_and_blur_decodings(
                    image=sharpened_gray, blur_kernel_sizes=((3, 3),)
                )
                if len(decodedQR) > 0:
                    return wrap(
                        scale_factor=scale_factor,
                        corrections=label,
                        flavor="grayscale",
                        blur_kernel_sizes=((3, 3),),
                        image=sharpened_gray,
                        results=decodedQR,
                    )

        return []

    def __correct_perspective(
        self, image: np.ndarray, padded_quad_xy: np.ndarray
    ) -> np.ndarray:
        """
        :param image: np.ndarray. The image to be read. It must be a np.ndarray (HxWxC) (uint8).
        :param padded_quad_xy: np.ndarray. An expanded version of quad_xy, with shape (4, 2), dtype: np.float32.
        :return: np.ndarray. The image with the perspective corrected.
        """

        # Define the width and height of the quadrilateral
        width1 = np.sqrt(
            ((padded_quad_xy[0][0] - padded_quad_xy[1][0]) ** 2)
            + ((padded_quad_xy[0][1] - padded_quad_xy[1][1]) ** 2)
        )
        width2 = np.sqrt(
            ((padded_quad_xy[2][0] - padded_quad_xy[3][0]) ** 2)
            + ((padded_quad_xy[2][1] - padded_quad_xy[3][1]) ** 2)
        )

        height1 = np.sqrt(
            ((padded_quad_xy[0][0] - padded_quad_xy[3][0]) ** 2)
            + ((padded_quad_xy[0][1] - padded_quad_xy[3][1]) ** 2)
        )
        height2 = np.sqrt(
            ((padded_quad_xy[1][0] - padded_quad_xy[2][0]) ** 2)
            + ((padded_quad_xy[1][1] - padded_quad_xy[2][1]) ** 2)
        )

        # Take the maximum width and height to ensure no information is lost
        max_width = max(int(width1), int(width2))
        max_height = max(int(height1), int(height2))
        N = max(max_width, max_height)

        # Create destination points for the perspective transform. This forms an N x N square
        dst_pts = np.array(
            [[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]], dtype=np.float32
        )

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(padded_quad_xy, dst_pts)

        # Perform the perspective warp
        dst_img = cv2.warpPerspective(image, M, (N, N))

        return dst_img

    def __threshold_and_blur_decodings(
        self, image: np.ndarray, blur_kernel_sizes: tuple[tuple[int, int]] = ((3, 3),)
    ) -> list[Decoded]:
        """
        Try to decode the QR code just with pyzbar, pre-processing the image with different blur and threshold
        filters.
        :param image: np.ndarray. The image to be read. It must be a 2D or 3D np.ndarray (HxW[xC]) (uint8).
        :return: list[Decoded]. The decoded QR code/s in the zbar format. If it fails, it will return an empty list.
        """

        assert (
            2 <= len(image.shape) <= 3
        ), f"image must be 2D or 3D (HxW[xC]) (uint8). Got {image.shape}"
        decodedQR = decodeQR(image=image, symbols=[ZBarSymbol.QRCODE])
        if len(decodedQR) > 0:
            return decodedQR

        # Try to binarize the image (Only works with 2D images)
        if len(image.shape) == 2:
            _, binary_image = cv2.threshold(
                image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            decodedQR = decodeQR(image=binary_image, symbols=[ZBarSymbol.QRCODE])
            if len(decodedQR) > 0:
                return decodedQR

        for kernel_size in blur_kernel_sizes:
            assert (
                isinstance(kernel_size, tuple) and len(kernel_size) == 2
            ), f"kernel_size must be a tuple of 2 elements. Got {kernel_size}"
            assert all(
                kernel_size[i] % 2 == 1 for i in range(2)
            ), f"kernel_size must be a tuple of odd elements. Got {kernel_size}"

            # If it not works, try to parse to sharpened grayscale
            blur_image = cv2.GaussianBlur(src=image, ksize=kernel_size, sigmaX=0)
            decodedQR = decodeQR(image=blur_image, symbols=[ZBarSymbol.QRCODE])
            if len(decodedQR) > 0:
                return decodedQR
        return []
