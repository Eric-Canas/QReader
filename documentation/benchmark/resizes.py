import cv2
import imageio
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode

from qreader import QReader

qreader_reader, cv2_reader, pyzbar_reader = (
    QReader(model_size="m"),
    cv2.QRCodeDetector(),
    decode,
)


def get_scaled_sizes(start_size, end_size, step, w, h):
    """Yield scaled sizes for given start and end size."""
    for size in range(start_size, end_size - 1, -step):
        scale = size / w
        yield (int(w * scale), int(h * scale))


def validate_and_write_on_image(image, current_size):

    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def draw_tick(x, y):
        cv2.line(image_copy, (x, y), (x + 5, y + 10), (0, 255, 0), 2)
        cv2.line(image_copy, (x + 5, y + 10), (x + 15, y - 10), (0, 255, 0), 2)

    def draw_cross(x, y):
        cv2.line(image_copy, (x, y), (x + 20, y + 20), (0, 0, 255), 2)
        cv2.line(image_copy, (x + 20, y), (x, y + 20), (0, 0, 255), 2)

    def draw_warn(x, y):
        cv2.putText(
            image_copy,
            "!",
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    qreader_out = qreader_reader.detect_and_decode(image=image)
    cv2_out = cv2_reader.detectAndDecode(image)[0]
    pyzbar_out = pyzbar_reader(image=image)

    qreader_status = (
        "YES"
        if len(qreader_out) > 0 and qreader_out[0] is not None
        else "WARN" if len(qreader_out) > 0 else "NO"
    )
    cv2_status = "YES" if cv2_out != "" else "NO"
    pyzbar_status = "YES" if len(pyzbar_out) > 0 else "NO"

    cv2.putText(
        image_copy,
        f"Size: {current_size}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    y_position = 80
    x_position_text = 20
    x_position_symbol = 220

    for method, status in [
        ("OpenCV", cv2_status),
        ("Pyzbar", pyzbar_status),
        ("QReader", qreader_status),
    ]:
        cv2.putText(
            image_copy,
            f"{method}:",
            (x_position_text, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        if status == "YES":
            draw_tick(x_position_symbol, y_position - 10)
        elif status == "NO":
            draw_cross(x_position_symbol, y_position - 10)
        elif status == "WARN":
            draw_warn(x_position_symbol, y_position - 10)

        y_position += 40

    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


def main():
    image = cv2.imread("../resources/logo.png", cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    frames = []

    for size in range(640, 10, -5):
        size_h = size_w = size
        resized_image = cv2.resize(
            image, (size_w, size_h), interpolation=cv2.INTER_AREA
        )

        pad_h = (640 - size_h) // 2
        pad_w = (640 - size_w) // 2
        resized_image = cv2.copyMakeBorder(
            resized_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255
        )

        resized_image = validate_and_write_on_image(resized_image, f"{size_w}x{size_h}")
        frames.append(Image.fromarray(resized_image))

        if (size_w % 50) == 0:
            print(f"Done {size_w}x{size_h}")

    gif_path = "resized_image.gif"
    imageio.mimsave(gif_path, frames, duration=0.1)


if __name__ == "__main__":
    main()
