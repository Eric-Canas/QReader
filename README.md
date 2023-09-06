# QReader

<img alt="QReader" title="QReader" src="https://raw.githubusercontent.com/Eric-Canas/QReader/main/documentation/resources/logo.png" width="20%" align="left"> **QReader** is a **Robust** and **Straight-Forward** solution for reading **difficult** and **tricky** **QR** codes within images in **Python**. Powered by a <a href="https://github.com/Eric-Canas/qrdet" target="_blank">YOLOv8</a> model.

Behind the scenes, the library is composed by two main building blocks: A <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLOv8</a> **QR Detector** model trained to **detect** and **segment** QR codes (also offered as <a href="https://github.com/Eric-Canas/qrdet" target="_blank">stand-alone</a>), and the <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a> **QR Decoder**. 

Using the information extracted from this **QR Detector**, **QReader** transparently applies, on top of <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a>, different image preprocessing techniques that maximize the **decoding** rate on difficult images.


## Installation

To install **QReader**, simply run:

```bash
pip install qreader
```

You may need to install some additional **pyzbar** dependencies:

On **Windows**:  

If you see an ugly ImportError related with `lizbar-64.dll`, install the [vcredist_x64.exe](https://www.microsoft.com/en-gb/download/details.aspx?id=40784) from the _Visual C++ Redistributable Packages for Visual Studio 2013_

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

# Get the image that contains the QR code (QReader expects an uint8 numpy array)
image = cv2.cvtColor(cv2.imread("path/to/image.png"), cv2.COLOR_BGR2RGB)

# Use the detect_and_decode function to get the decoded QR data
decoded_text = qreader.detect_and_decode(image=image)
```

``detect_and_decode`` will return a `tuple` containing the decoded _string_ of every **QR** found in the image. 
 **NOTE**: Some entries can be `None`, it will happen when a **QR** have been detected but **couldn't be decoded**.


## API Reference

### QReader(reencode_to = 'shift-jis')

This is the main class of the library. Please, try to instantiate it just once to avoid loading the model every time 
you need to detect a **QR** code.
- ``reencode_to``: **str** or **None**. The encoding to reencode the `utf-8` decoded QR string. If None, it won't re-encode. If you find some characters being decoded incorrectly, try to set a [Code Page](https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers) that matches your specific charset. Recommendations that have been found useful:
  - 'shift-jis' for Germanic languages
  - 'cp65001' for Asian languages (Thanks to @nguyen-viet-hung for the suggestion)

### QReader.detect_and_decode(image, return_bboxes = False)

This method will decode the **QR** codes in the given image and return the decoded _strings_ (or None, if any of them could be detected but not decoded).

- ``image``: **np.ndarray**. NumPy Array containing the ``image`` to decode. The image is expected to be in ``uint8`` format [_HxWxC_], RGB.
- ``return_bboxes``: **boolean**. If ``True``, it will also return the bboxes of each detected **QR**. Default: `False`


- Returns: **tuple[str | None] | tuple[tuple[tuple[int, int, int, int], str | None]]**: A tuple with all detected **QR** codes decodified. If ``return_bboxes`` is `False`, the output will look like: `('Decoded QR 1', 'Decoded QR 2', None, 'Decoded QR 4', ...)`. If ``return_bboxes`` is `True` it will look like: `(((x1_1, y1_1, x2_1, y2_1), 'Decoded QR 1'), ((x1_2, y1_2, x2_2, y2_2), 'Decoded QR 2'), ...)`.

### QReader.detect(image)

This method detects the **QR** codes in the image and returns the **bounding boxes** surrounding them in the format (_x1_, _y1_, _x2_, _y2_). 

- ``image``: **np.ndarray**. NumPy Array containing the ``image`` to decode. The image must is expected to be in ``uint8`` format [_HxWxC_], RGB.


- Returns: **tuple[tuple[int, int, int, int]]**. The bounding boxes of the **QR** code in the format `((x1_1, y1_1, x2_1, y2_1), (x1_1, y1_1, x2_1, x2_2))`.

**NOTE**: This the only function you will need? Take a look at <a href="https://github.com/Eric-Canas/qrdet" target="_blank">QRDet</a>.

### QReader.decode(image, bbox = None)

This method decodes a single **QR** code on the given image, if a ``bbox`` is given (recommended) it will only look within that delimited region.

Internally, this method will run the <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">pyzbar</a> decoder, using different image preprocessing techniques (_sharpening_, _binarization_, _blurring_...) every time it fails to increase the detection rate.

- ``image``: **np.ndarray**. NumPy Array containing the ``image`` to decode. The image must is expected to be in ``uint8`` format [_HxWxC_], RGB.
- ``bbox``: **tuple[int, int, int, int] | None**. The bounding box of the **QR** code in the format (_x1_, _y1_, _x2_, _y2_) [that's the output of `detect`]. If ``None``, it will look for the **QR** code in the whole image (not recommended). Default: ``None``.


- Returns: **str**. The decoded text of the **QR** code. If no **QR** code can be decoded, it will return ``None``.

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

Note that **QReader** internally uses <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">pyzbar</a> as **decoder**. The improved **detection-decoding rate** that **QReader** achieves comes from the combination of different image pre-processing techniques and the <a href="https://github.com/WongKinYiu/yolov7" target="_blank">YOLOv7</a> based <a href="https://github.com/Eric-Canas/qrdet" target="_blank">**QR** detector</a> that is able to detect **QR** codes in harder conditions than classical _Computer Vision_ methods.

## Acknowledgements

This library is based on the following projects:

- <a href="https://github.com/WongKinYiu/yolov7" target="_blank">YoloV7</a> model for **Object Detection**.
- <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a> **QR** Decoder.
- <a href="https://opencv.org/" target="_blank">OpenCV</a> methods for image filtering.
