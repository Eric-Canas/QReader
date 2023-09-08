# QReader

<img alt="QReader" title="QReader" src="https://raw.githubusercontent.com/Eric-Canas/QReader/main/documentation/resources/logo.png" width="20%" align="left"> **QReader** is a **Robust** and **Straight-Forward** solution for reading **difficult** and **tricky** **QR** codes within images in **Python**. Powered by a <a href="https://github.com/Eric-Canas/qrdet" target="_blank">YOLOv8</a> model.

Behind the scenes, the library is composed by two main building blocks: A <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLOv8</a> **QR Detector** model trained to **detect** and **segment** QR codes (also offered as <a href="https://github.com/Eric-Canas/qrdet" target="_blank">stand-alone</a>), and the <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a> **QR Decoder**. Using the information extracted from this **QR Detector**, **QReader** transparently applies, on top of <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a>, different image preprocessing techniques that maximize the **decoding** rate on difficult images.


## Installation

To install **QReader**, simply run:

```bash
pip install qreader
```

You may need to install some additional **pyzbar** dependencies:

On **Windows**:  

Rarely, you can see an ugly ImportError related with `lizbar-64.dll`. If it happens, install the [vcredist_x64.exe](https://www.microsoft.com/en-gb/download/details.aspx?id=40784) from the _Visual C++ Redistributable Packages for Visual Studio 2013_

On **Linux**:  
```bash
sudo apt-get install libzbar0
```

On **Mac OS X**: 
```bash
brew install zbar
```

**NOTE:** If you're running **QReader** in a server with very limited resources, you may want to install the **CPU** version of [**PyTorch**](https://pytorch.org/get-started/locally/), before installing **QReader**. To do so, run: ``pip install torch --no-cache-dir`` (Thanks to [**@cjwalther**](https://github.com/Eric-Canas/QReader/issues/5) for his advice).

## Usage
<a href="https://colab.research.google.com/github/Eric-Canas/QReader/blob/main/example.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

**QReader** is a very simple and straight-forward library. For most use cases, you'll only need to call ``detect_and_decode``:

```python
from qreader import QReader
import cv2


# Create a QReader instance
qreader = QReader()

# Get the image that contains the QR code
image = cv2.cvtColor(cv2.imread("path/to/image.png"), cv2.COLOR_BGR2RGB)

# Use the detect_and_decode function to get the decoded QR data
decoded_text = qreader.detect_and_decode(image=image)
```

``detect_and_decode`` will return a `tuple` containing the decoded _string_ of every **QR** found in the image. 

> **NOTE**: Some entries can be `None`, it will happen when a **QR** have been detected but **couldn't be decoded**.


## API Reference

### QReader(model_size = 's', min_confidence = 0.5, reencode_to = 'shift-jis')

This is the main class of the library. Please, try to instantiate it just once to avoid loading the model every time you need to detect a **QR** code.
- ``model_size``: **str**. The size of the model to use. It can be **'n'** (nano), **'s'** (small), **'m'** (medium) or **'l'** (large). Larger models are more accurate but slower. Default: 's'.
- ``min_confidence``: **float**. The minimum confidence of the QR detection to be considered valid. Values closer to 0.0 can get more _False Positives_, while values closer to 1.0 can lose difficult QRs. Default (and recommended): 0.5.
- ``reencode_to``: **str** | **None**. The encoding to reencode the `utf-8` decoded QR string. If None, it won't re-encode. If you find some characters being decoded incorrectly, try to set a [Code Page](https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers) that matches your specific charset. Recommendations that have been found useful:
  - 'shift-jis' for Germanic languages
  - 'cp65001' for Asian languages (Thanks to @nguyen-viet-hung for the suggestion)

### QReader.detect_and_decode(image, return_detections = False)

This method will decode the **QR** codes in the given image and return the decoded _strings_ (or _None_, if any of them was detected but not decoded).

- ``image``: **np.ndarray**. The image to be read. It is expected to be _RGB_ or _BGR_ (_uint8_). Format (_HxWx3_).
- ``return_detections``: **bool**. If `True`, it will return the full detection results together with the decoded QRs. If False, it will return only the decoded content of the QR codes.
- ``is_bgr``: **boolean**. If `True`, the received image is expected to be _BGR_ instead of _RGB_.

  
- **Returns**: **tuple[str | None] | tuple[tuple[dict[str, np.ndarray | float | tuple[float | int, float | int]]], str | None]]**: A tuple with all detected **QR** codes decodified. If ``return_detections`` is `False`, the output will look like: `('Decoded QR 1', 'Decoded QR 2', None, 'Decoded QR 4', ...)`. If ``return_detections`` is `True` it will look like: `(('Decoded QR 1', {'bbox_xyxy': (x1_1, y1_1, x2_1, y2_1), 'confidence': conf_1}), ('Decoded QR 2', {'bbox_xyxy': (x1_2, y1_2, x2_2, y2_2), 'confidence': conf_2, ...}), ...)`. Look [QReader.detect()](#QReader_detect_table) for more information about detections format.

<a name="QReader_detect"></a>

### QReader.detect(image)

This method detects the **QR** codes in the image and returns a _tuple of dictionaries_ with all the detection information.

- ``image``: **np.ndarray**. The image to be read. It is expected to be _RGB_ or _BGR_ (_uint8_). Format (_HxWx3_).
- ``is_bgr``: **boolean**. If `True`, the received image is expected to be _BGR_ instead of _RGB_.
<a name="QReader_detect_table"></a>

- **Returns**: **tuple[dict[str, np.ndarray|float|tuple[float|int, float|int]]]**. A tuple of dictionaries containing all the information of every detection. Contains the following keys.

| Key              | Value Desc.                                 | Value Type                 | Value Form                  |
|------------------|---------------------------------------------|----------------------------|-----------------------------|
| `confidence`     | Detection confidence                        | `float`                    | `conf.`                     |
| `bbox_xyxy`      | Bounding box                                | np.ndarray (**4**)         | `[x1, y1, x2, y2]`          |
| `cxcy`           | Center of bounding box                      | tuple[`float`, `float`]    | `(x, y)`                    |
| `wh`             | Bounding box width and height               | tuple[`float`, `float`]    | `(w, h)`                    |
| `polygon_xy`     | Precise polygon that segments the _QR_      | np.ndarray (**N**, **2**)  | `[[x1, y1], [x2, y2], ...]` |
| `quad_xy`        | Four corners polygon that segments the _QR_ | np.ndarray (**4**, **2**)  | `[[x1, y1], ..., [x4, y4]]` |
| `padded_quad_xy` |`quad_xy` padded to fully cover `polygon_xy` | np.ndarray (**4**, **2**)  | `[[x1, y1], ..., [x4, y4]]` |
| `image_shape`    | Shape of the input image                    | tuple[`int`, `int`]    | `(h, w)`                    |  

> **NOTE:**
> - All `np.ndarray` values are of type `np.float32` 
> - All keys (except `confidence` and `image_shape`) have a normalized ('n') version. For example,`bbox_xyxy` represents the bbox of the QR in image coordinates [[0., im_w], [0., im_h]], while `bbox_xyxyn` contains the same bounding box in normalized coordinates [0., 1.].
> - `bbox_xyxy[n]` and `polygon_xy[n]` are clipped to `image_shape`. You can use them for indexing without further management


**NOTE**: Is this the only method you will need? Take a look at <a href="https://github.com/Eric-Canas/qrdet" target="_blank">QRDet</a>.

### QReader.decode(image, detection_result)

This method decodes a single **QR** code on the given image, described by a detection result. 

Internally, this method will run the <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">pyzbar</a> decoder, using the information of the `detection_result`, to apply different image preprocessing techniques that heavily increase the detecoding rate.

- ``image``: **np.ndarray**. NumPy Array with the ``image`` that contains the _QR_ to decode. The image is expected to be in ``uint8`` format [_HxWxC_], RGB.
- ``detection_result``: dict[str, np.ndarray|float|tuple[float|int, float|int]]. One of the **detection dicts** returned by the **detect** method. Note that [QReader.detect()](#QReader_detect) returns a `tuple` of these `dict`. This method expects just one of them.


- Returns: **str | None**. The decoded content of the _QR_ code or `None` if it couldn't be read.

## Usage Tests
<div><img alt="test_on_mobile" title="test_on_mobile" src="https://raw.githubusercontent.com/Eric-Canas/QReader/main/documentation/resources/test_mobile.jpeg" width="60%"><img alt="" title="QReader" src="https://raw.githubusercontent.com/Eric-Canas/QReader/main/documentation/resources/test_draw_64x64.jpeg" width="32%" align="right"></div>
<div>Two sample images. At left, an image taken with a mobile phone. At right, a 64x64 <b>QR</b> pasted over a drawing.</div>    
<br>

The following code will try to decode these images containing <b>QR</b>s with **QReader**, <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">pyzbar</a> and <a href="https://opencv.org/" target="_blank">OpenCV</a>.
```python
from qreader import QReader
from cv2 import QRCodeDetector, imread
from pyzbar.pyzbar import decode

# Initialize the three tested readers (QRReader, OpenCV and pyzbar)
qreader_reader, cv2_reader, pyzbar_reader = QReader(), QRCodeDetector(), decode

for img_path in ('test_mobile.jpeg', 'test_draw_64x64.jpeg'):
    # Read the image
    img = imread(img_path)

    # Try to decode the QR code with the three readers
    qreader_out = qreader_reader.detect_and_decode(image=img)
    cv2_out = cv2_reader.detectAndDecode(img=img)[0]
    pyzbar_out = pyzbar_reader(image=img)
    # Read the content of the pyzbar output (double decoding will save you from a lot of wrongly decoded characters)
    pyzbar_out = tuple(out.data.data.decode('utf-8').encode('shift-jis').decode('utf-8') for out in pyzbar_out)

    # Print the results
    print(f"Image: {img_path} -> QReader: {qreader_out}. OpenCV: {cv2_out}. pyzbar: {pyzbar_out}.")
```

The output of the previous code is:

```txt
Image: test_mobile.jpeg -> QReader: ('https://github.com/Eric-Canas/QReader'). OpenCV: . pyzbar: ().
Image: test_draw_64x64.jpeg -> QReader: ('https://github.com/Eric-Canas/QReader'). OpenCV: . pyzbar: ().
```

Note that **QReader** internally uses <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">pyzbar</a> as **decoder**. The improved **detection-decoding rate** that **QReader** achieves comes from the combination of different image pre-processing techniques and the <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLOv8</a> based <a href="https://github.com/Eric-Canas/qrdet" target="_blank">**QR** detector</a> that is able to detect **QR** codes in harder conditions than classical _Computer Vision_ methods.

## Benchmark

### Rotation Test
<div>
<img alt="Rotation Test" title="Rotation Test" src="https://raw.githubusercontent.com/Eric-Canas/QReader/main/documentation/benchmark/rotation_benchmark.gif" width="40%" align="left">

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; 
<div align="center">
  
| Method  | Max Rotation Degrees  |
|---------|-----------------------|
| Pyzbar  | 17º                   |
| OpenCV  | 46º                   |
| QReader | 79º                   |
  
</div>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; 
</div>

## Acknowledgements

This library is based on the following projects:

- <a href="https://github.com/ultralytics/ultralytics" target="_blank">YoloV8</a> model for **Object Detection & Segmentation**.
- <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a> **QR** Decoder.
- <a href="https://opencv.org/" target="_blank">OpenCV</a> methods for image filtering.
