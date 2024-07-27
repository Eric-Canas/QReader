import os
from time import time

import cv2
import numpy as np
from tqdm import tqdm

from qreader import QReader

SAMPLE_IMG_1 = os.path.join(
    os.path.dirname(__file__), "..", "..", "documentation", "resources", "64x64.png"
)
SAMPLE_IMG_2 = os.path.join(
    os.path.dirname(__file__), "..", "..", "documentation", "resources", "512x512.jpeg"
)
SAMPLE_IMG_3 = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "documentation",
    "resources",
    "1024x1024.jpeg",
)

PERFORMANCE_TEST_IAMGES = {
    "64x64": SAMPLE_IMG_1,
    #'512x512': SAMPLE_IMG_2,
    #'1024x1024': SAMPLE_IMG_3
}
RUNS_TO_AVERAGE, WARMUP_ITERATIONS = 5, 5


def test_performance():
    results = {}
    for shape, img_path in tqdm(PERFORMANCE_TEST_IAMGES.items()):
        # Read the image
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # Run the performance test over 100 iterations (warm-up included)
        detect_times, detect_and_decode_times = [], []
        for _ in range(RUNS_TO_AVERAGE + WARMUP_ITERATIONS):
            start = time()
            QReader().detect(image=img)
            detect_times.append(time() - start)
            start = time()
            QReader().detect_and_decode(image=img)
            detect_and_decode_times.append(time() - start)
        # Save the results
        results[shape] = {
            "detect": np.mean(detect_times[WARMUP_ITERATIONS:]),
            "detect_and_decode": np.mean(detect_and_decode_times[WARMUP_ITERATIONS:]),
        }
    # Print the results
    print("Performance test results:")
    for shape, times in results.items():
        print(
            f"Image shape: {shape} -> Detect: {times['detect']}. Detect and decode: {times['detect_and_decode']}."
        )
