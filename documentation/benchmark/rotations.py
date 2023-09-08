import numpy as np
import imageio
from PIL import Image

from qreader import QReader
import cv2
from pyzbar.pyzbar import decode

qreader_reader, cv2_reader, pyzbar_reader = QReader(model_size='m'), cv2.QRCodeDetector(), decode


def get_matrices(start_deg, end_deg, y_deg=20, w=256, h=256):
    """Yield rotation matrices for given start and end degrees, width, and height."""
    # 10 degrees in radians for Y-axis rotation

    rot_xs = np.linspace(np.radians(start_deg), np.radians(end_deg), end_deg - start_deg + 1)
    rot_ys = np.linspace(np.radians(0), np.radians(y_deg), end_deg - start_deg +1)
    # Iterate on radians. Rotate on X degree by degree
    for rotx, roty in zip(rot_xs, rot_ys):
        focal_length = 2
        cosx, sinx = np.cos(rotx), np.sin(rotx)

        # X-axis rotation matrix
        rotox = [
            [1, 0, 0],
            [0, cosx, -sinx],
            [0, sinx, cosx]
        ]
        cosy, siny = np.cos(roty), np.sin(roty)

        # Y-axis rotation matrix
        roty = [
            [cosy, 0, siny],
            [0, 1, 0],
            [-siny, 0, cosy]
        ]

        # Combine X and Y rotation matrices
        combined_roto = np.dot(rotox, roty)

        pt = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                       [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float32)
        ptt = np.dot(pt, np.transpose(combined_roto))

        # Generate input and output points
        ptt[:, 0] = w / 2 + ptt[:, 0] * focal_length * h / (focal_length * h + ptt[:, 2])
        ptt[:, 1] = h / 2 + ptt[:, 1] * focal_length * h / (focal_length * h + ptt[:, 2])

        in_pt = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        out_pt = np.array(ptt[:, :2], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(in_pt, out_pt)
        yield transform_matrix



def validate_and_write_on_image(image, current_degs_x, current_degs_y):
    def draw_tick(x, y):
        # Short line (almost vertical)
        cv2.line(image_copy, (x, y), (x + 5, y + 10), (0, 255, 0), 2)

        # Long line (more inclined)
        cv2.line(image_copy, (x + 5, y + 10), (x + 15, y - 10), (0, 255, 0), 2)

    def draw_cross(x, y):
        cv2.line(image_copy, (x, y), (x + 20, y + 20), (0, 0, 255), 2)
        cv2.line(image_copy, (x + 20, y), (x, y + 20), (0, 0, 255), 2)

    def draw_warn(x, y):
        cv2.putText(image_copy, "!", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def overlay_transparent_rect(image, x, y, width, height, color, alpha):
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Create a copy of the image and convert it to a 3-channel image for color
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Try to decode the QR code with the three readers
    qreader_out = qreader_reader.detect_and_decode(image=image)
    cv2_out = cv2_reader.detectAndDecode(image)[0]
    pyzbar_out = pyzbar_reader(image=image)

    # Create status indicators based on decoding success
    qreader_status = "YES" if len(qreader_out) > 0 and qreader_out[0] is not None else "WARN" if len(
        qreader_out) > 0 else "NO"
    cv2_status = "YES" if cv2_out != "" else "NO"
    pyzbar_status = "YES" if len(pyzbar_out) > 0 else "NO"

    overlay_transparent_rect(image_copy, 10, 0, 500, 40, (255, 255, 255), 0.6)
    overlay_transparent_rect(image_copy, 10, 50, 250, 40*3+20, (255, 255, 255), 0.6)
    # Writing the degrees in the top left corner
    cv2.putText(image_copy, f" x: {current_degs_x} deg - y: {round(current_degs_y, 1)} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 2)

    # Status positions
    y_position = 80
    y_increment = 40
    x_position_text = 20
    x_position_symbol = 220

    for method, status in [("OpenCV", cv2_status), ("Pyzbar", pyzbar_status), ("QReader", qreader_status)]:
        # Draw semi-transparent white rectangle as a background

        # Draw text in black
        cv2.putText(image_copy, f"   {method}:", (x_position_text, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw the symbol next to the text
        if status == "YES":
            draw_tick(x_position_symbol, y_position - 10)
        elif status == "NO":
            draw_cross(x_position_symbol, y_position - 10)
        elif status == "WARN":
            draw_warn(x_position_symbol, y_position - 10)

        y_position += y_increment  # Move down for the next line

    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)




def main():
    # Load an image (replace 'your_image_path.png' with the path to your image)
    image = cv2.imread('../resources/logo.png', cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    frames = []

    # Get the max transformation matrix
    m = next(get_matrices(start_deg=80-1, end_deg=80, w=w, h=h))
    # Apply the perspective transformation for the corners
    im_corners = cv2.perspectiveTransform(np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32), m)
    # Get each side pad
    left_pad = abs(int(np.min(im_corners[:, :, 0])))
    right_pad = int(np.max(im_corners[:, :, 0])) - w
    # Apply the pads to the image
    image = np.pad(image, ((10, 0), (left_pad-10, right_pad-10)), mode='constant', constant_values=255)
    h, w = image.shape
    degs = 0
    degs_y = 0
    for matrix in get_matrices(start_deg=0, end_deg=80, y_deg=25, w=w, h=h):
        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # Validate and write on the image
        transformed_image = validate_and_write_on_image(transformed_image, degs, degs_y)
        frames.append(Image.fromarray(transformed_image))
        degs += 1
        degs_y += 25/80
        if (degs % 10) == 0:
            print(f"Done {degs}º")

    # Save the GIF
    gif_path = 'rotated_image.gif'
    imageio.mimsave(gif_path, frames, duration=0.1)


if __name__ == '__main__':
    main()
