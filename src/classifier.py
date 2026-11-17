from pathlib import Path

import cv2
import joblib
import numpy as np

from src.config import MODEL_PATH


class VehicleClassifier:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Brak modelu {model_path}. Najpierw uruchom: python -m src.train")
        self.model = joblib.load(model_path)

    def predict_image(self, image):
        return str(self.model.predict([image])[0])

    def predict(self, image_path):
        # np.fromfile zamiast cv2.imread, bo imread na Windows nie czyta ścieżek z polskimi znakami
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Plik nie jest obrazem: {image_path}")
        return self.predict_image(image)