from functools import lru_cache

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
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

# Średnie i odchylenia kanałów RGB ze zbioru ImageNet, na którym uczono MobileNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def load_mobilenet():
    """MobileNetV2 z wagami z ImageNet, bez ostatniej warstwy klasyfikującej."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    # Zamiast 1000 klas ImageNet sieć zwraca 1280 cech z global average pooling
    model.classifier = torch.nn.Identity()
    return model.eval()


def to_mobilenet_input(image, size):
    # Cały wycinek skalujemy do kwadratu jak w HOG. Środkowe przycięcie
    # z torchvision ucięłoby przód i tył długich pojazdów (Bus, Truck).
    rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)  # HWC -> CHW, kolejność wymiarów w PyTorch


class MobileNetTransformer(BaseEstimator, TransformerMixin):
    """Cechy z gotowej sieci MobileNetV2. Sieć nie jest dotrenowywana, uczy się tylko SVM."""

    def __init__(self, size=224, batch_size=32):
        self.size = size
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = load_mobilenet()
        features = []
        with torch.inference_mode():
            for start in range(0, len(X), self.batch_size):
                batch = np.stack([to_mobilenet_input(image, self.size) for image in X[start:start + self.batch_size]])
                features.append(model(torch.from_numpy(batch)).numpy())
        return np.concatenate(features)