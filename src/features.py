import cv2
import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin


def to_gray_square(image, size):
    resized = cv2.resize(image, (size, size))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


class HogTransformer(BaseEstimator, TransformerMixin):
    # Domyślne wartości dobrane walidacją krzyżową na train + valid.
    # Przy 128 px, komórkach 16 px i 12 kierunkach HOG daje 2352 cechy.
    def __init__(self, size=128, orientations=12, pixels_per_cell=16, cells_per_block=2):
        self.size = size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(image) for image in X])

    def _extract(self, image):
        return hog(
            to_gray_square(image, self.size),
            orientations=self.orientations,
            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
            cells_per_block=(self.cells_per_block, self.cells_per_block),
            block_norm="L2-Hys",
        )


class PixelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, size=32):
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([to_gray_square(image, self.size).ravel() for image in X], dtype=np.float32)