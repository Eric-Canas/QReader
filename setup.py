from setuptools import setup, find_namespace_packages

setup(
    name='qreader',
    version='3.1',
    packages=find_namespace_packages(),
    # expose qreader.py as the unique module
    py_modules=['qreader'],
    url='https://github.com/Eric-Canas/qreader',
    license='MIT',
    author='Eric Canas',
    author_email='elcorreodeharu@gmail.com',
    description='Robust and Straight-Forward solution for reading difficult and tricky QR codes '
                'within images in Python. Supported by a YOLOv8 QR Segmentation model.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'opencv-python',
        'pyzbar',
        'qrdet>=2.1',
    ],
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
