"""
TF-explain Library

The library implements interpretability methods as Tensorflow 2.0
callbacks to ease neural network's understanding.
"""

__version__ = "0.3.0"

try:
    import cv2
except:
    raise ImportError(
        "TF-explain requires Opencv. " "Install Opencv via `pip install opencv-python`"
    ) from None
try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TF-explain requires TensorFlow 2.0 or higher. "
        "Install TensorFlow via `pip install tensorflow`"
    ) from None
from . import core
from . import callbacks
from . import utils
