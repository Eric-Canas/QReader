# QReader

QReader is a <b>Robust</b> and <b>Straight-Forward</b> solution for reading <b>difficult</b> and <b>tricky</b> QR codes within images in <a href=https://www.python.org/ target="_blank"><img alt="Python" title="Python" src="https://img.shields.io/static/v1?label=&message=Python&color=3C78A9&logo=python&logoColor=FFFFFF"></a>.

Behind the scenes, this detector is based on several other **Detectors** & **Decoders**, such as <a href="https://github.com/NaturalHistoryMuseum/pyzbar" target="_blank">Pyzbar</a>, <a href="https://opencv.org/" target="_blank">OpenCV</a> and <a href="https://github.com/Gbellport/QR-code-localization-YOLOv3" target="_blank">YoloV3</a>, as well as different image preprocessing techniques. **QReader** will transparently combine all these techniques to maximize the detection rate on difficult images (e.g. QR code too small).

## Installation

To install QReader, simply run:

```bash
pip install qreader
```

If you're not using **Windows**, you might need to install some additional **pyzbar** dependencies:

On Linux:
```bash
sudo apt-get install libzbar0
```

On Mac OS X:
```bash
brew install zbar
```
