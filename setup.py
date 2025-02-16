from setuptools import find_packages, setup

setup(
    name="qreader",
    version="3.15b0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # expose qreader.py as the unique module
    py_modules=["qreader"],
    # include py.typed file in the distribution
    package_data={"qreader": ["src/qreader/py.typed"]},
    url="https://github.com/Eric-Canas/qreader",
    license="MIT",
    author="Eric Canas",
    author_email="elcorreodeharu@gmail.com",
    description="Robust and Straight-Forward solution for reading difficult and tricky QR codes "
    "within images in Python. Supported by a YOLOv8 QR Segmentation model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "opencv-python",
        "pyzbar",
        "qrdet>=2.5",
    ],
    extras_require={
        "tests": ["mypy", "pytest", "qrcode"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
)
