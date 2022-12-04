from setuptools import setup, find_namespace_packages

setup(
    name='qreader',
    version='1.2.3',
    packages=find_namespace_packages(),
    # expose qreader.py as the unique module
    py_modules=['qreader'],
    url='https://github.com/Eric-Canas/qreader',
    license='MIT',
    author='Eric Canas',
    author_email='elcorreodeharu@gmail.com',
    description='Robust and Straight-Forward solution for reading difficult and tricky QR codes within images in Python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'opencv-python',
        'pyzbar',
    ],
    # To include the __yolo_v3_qr_detector weights in the package, we need to add the following line:
    include_package_data=True,
    # To include the __yolo_v3_qr_detector weights in the package, we need to add the following line:
    data_files=[('__yolo_v3_qr_detector',
                 ['__yolo_v3_qr_detector/qrcode-yolov3-tiny.cfg',
                  '__yolo_v3_qr_detector/qrcode-yolov3-tiny_last.weights'])],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
)
