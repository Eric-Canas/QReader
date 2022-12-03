"""
This class implements a YoloV3 detector of QR codes.
It uses the YoloV3 models pre-trained by Gabriel Bellport for QR code localization
(https://github.com/Gbellport/QR-code-localization-YOLOv3)

Mail: eric@ericcanas.com
Date: 03-12-2022
Github: https://github.com/Eric-Canas
"""

from __future__ import annotations

import cv2
from yolo_v3.constants import YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH
import numpy as np

_INPUT_SIZE = (416, 416)
_CONF_THRESHOLD = 0.5

class _YoloV3QRDetector:
    def __init__(self):
        """
        This class uses the YoloV3 models pre-trained by Gabriel Bellport for QR code localization
        (https://github.com/Gbellport/QR-code-localization-YOLOv3)
        """

        # YOLOV3 based QR detector for the difficult cases (quite ugly code, but it is what it is)
        self.yolo_v3_QR_detector = cv2.dnn.readNetFromDarknet(cfgFile=YOLO_CONFIG_PATH, darknetModel=YOLO_WEIGHTS_PATH)
        self.yolo_v3_QR_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        output_layer_names = self.yolo_v3_QR_detector.getLayerNames()
        self.output_layer_name = output_layer_names[self.yolo_v3_QR_detector.getUnconnectedOutLayers()[0] - 1]


    def detect(self, img: np.ndarray)-> tuple[bool, tuple[int, int, int, int]|None]:
        """
        This method will detect the QR code in the image. If the QR code is detected, it will return True and the bounding
        box of the QR code. If the QR code is not detected, it will return False and None.

        :param img: np.ndarray. The image where the QR code will be detected, as an uint8 numpy array (HxWxC).

        :return: tuple[bool, tuple[int, int, int, int]|None]. The first element is a boolean indicating if the QR code
                                                              was detected or not. The second element is the bounding
                                                              box of the QR code, if it was detected or None otherwise.
        """
        # Transform the image to blob and predict
        blob = cv2.dnn.blobFromImage(img, 1 / 255, _INPUT_SIZE, swapRB=False, crop=False)
        self.yolo_v3_QR_detector.setInput(blob=blob)
        output = self.yolo_v3_QR_detector.forward(self.output_layer_name)

        # Post-process the output
        h, w = img.shape[:2]
        qr_bbox = self.__postprocess(h=h, w=w, out=output)
        return qr_bbox is not None, qr_bbox

    def __postprocess(self, h: int, w: int, out, threshold=_CONF_THRESHOLD) -> tuple[int, int, int, int] | None:
        """
        This method will post-process the output of the YoloV3 model to obtain the bounding box of the QR code.

        :param h: int. The original height of the image.
        :param w: int. The original width of the image.
        :param out: np.ndarray. The output of the YoloV3 model.
        :param threshold: float. The confidence threshold to consider a detection as valid. Default: 0.5.

        :return: tuple[int, int, int, int] | None. The bounding box of the QR code, if it was detected. None otherwise.
        """

        # Return only the best detection
        valid_bboxes = out[out[..., -1] > threshold]
        if len(valid_bboxes) > 0:
            best_bbox_idx = np.argmax(valid_bboxes[..., -1])
            x, y, width, height = valid_bboxes[best_bbox_idx][:4]*(w, h, w, h)
            # Yolo gives the center of the box, so we need to convert it to the top left corner
            x_1, y_1 = max(int(x - width / 2), 0), max(int(y - height / 2), 0)
            x_2, y_2 = int(x + width / 2), int(y + height / 2)

            return (x_1, y_1, x_2, y_2)
        else:
            return None



