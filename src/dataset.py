from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import SPLITS

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# Roboflow dopisuje do nazwy pliku ".rf.<hash>", np. "0a1b2c_jpg.rf.9f8e7d.jpg".
# To samo zdjęcie źródłowe może wystąpić kilka razy z różnymi hashami.
ROBOFLOW_MARKER = ".rf."
# Separator między nazwą zdjęcia źródłowego a numerem ramki w nazwie wycinka
CROP_SEPARATOR = "__"


def read_image(path):
    # np.fromfile zamiast cv2.imread, bo imread na Windows nie czyta ścieżek z polskimi znakami
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def source_name(image_path):
    """Nazwa zdjęcia źródłowego bez dopisku Roboflow, wspólna dla duplikatów."""
    name = Path(image_path).name
    return name.split(ROBOFLOW_MARKER)[0] if ROBOFLOW_MARKER in name else Path(name).stem


def crop_name(source, index):
    return f"{source}{CROP_SEPARATOR}{index}.jpg"


def source_of_crop(crop_path):
    return Path(crop_path).stem.rsplit(CROP_SEPARATOR, 1)[0]


@dataclass(frozen=True)
class BoundingBox:
    """Ramka w formacie YOLO: środek, szerokość i wysokość jako ułamki wymiarów zdjęcia."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_yolo_line(cls, line):
        values = line.split()
        class_id = int(values[0])
        coords = [float(v) for v in values[1:]]
        if len(coords) == 4:
            return cls(class_id, *coords)
        if len(coords) >= 6 and len(coords) % 2 == 0:
            # Etykieta segmentacji (wielokąt x1 y1 x2 y2 ...): bierzemy ramkę otaczającą
            xs, ys = coords[0::2], coords[1::2]
            return cls(
                class_id,
                (min(xs) + max(xs)) / 2,
                (min(ys) + max(ys)) / 2,
                max(xs) - min(xs),
                max(ys) - min(ys),
            )
        raise ValueError(f"Niepoprawna linia etykiety YOLO: {line!r}")

    def to_pixels(self, image_width, image_height):
        """Zwraca (x1, y1, x2, y2) w pikselach, przycięte do granic zdjęcia."""
        x1 = round((self.x_center - self.width / 2) * image_width)
        y1 = round((self.y_center - self.height / 2) * image_height)
        x2 = round((self.x_center + self.width / 2) * image_width)
        y2 = round((self.y_center + self.height / 2) * image_height)
        return (
            max(0, x1),
            max(0, y1),
            min(image_width, x2),
            min(image_height, y2),
        )

    def short_side(self, image_width, image_height):
        x1, y1, x2, y2 = self.to_pixels(image_width, image_height)
        return min(x2 - x1, y2 - y1)

    def crop(self, image):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self.to_pixels(width, height)
        return image[y1:y2, x1:x2]


class YoloDataset:
    """Zbiór w formacie YOLOv8 z Roboflow: data.yaml oraz <split>/images i <split>/labels."""

    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        yaml_path = self.raw_dir / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Brak {yaml_path}. Rozpakuj zbiór z Roboflow do {self.raw_dir}")
        with open(yaml_path, encoding="utf-8") as file:
            names = yaml.safe_load(file)["names"]
        # names bywa listą albo słownikiem {id: nazwa}
        self.class_names = list(names.values()) if isinstance(names, dict) else list(names)
        self.skipped_duplicates = []

    def images(self, splits=SPLITS):
        """Zwraca (split, ścieżka zdjęcia, ramki), pomijając duplikaty zdjęć.

        Duplikat to zdjęcie o tej samej nazwie źródłowej co zdjęcie już zwrócone.
        Zostaje pierwsze wystąpienie w kolejności splitów, więc duplikat z valid
        lub test zdjęcia z train jest pomijany i nie przecieka między zbiorami.
        """
        seen = set()
        self.skipped_duplicates = []
        for split in splits:
            images_dir = self.raw_dir / split / "images"
            labels_dir = self.raw_dir / split / "labels"
            if not images_dir.exists():
                continue
            for image_path in sorted(images_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                source = source_name(image_path)
                if source in seen:
                    self.skipped_duplicates.append((split, image_path))
                    continue
                seen.add(source)
                yield split, image_path, self.read_labels(labels_dir / f"{image_path.stem}.txt")

    @staticmethod
    def read_labels(label_path):
        if not label_path.exists():
            return []
        with open(label_path, encoding="utf-8") as file:
            return [BoundingBox.from_yolo_line(line) for line in file if line.strip()]

    def class_name(self, class_id):
        return self.class_names[class_id]


def load_crops(crops_dir, splits):
    """Wczytuje wycinki z <crops_dir>/<split>/<klasa>/*.jpg.

    Zwraca (images, labels, groups): obrazy BGR, nazwy klas i nazwy zdjęć źródłowych.
    Grupy pilnują, żeby wycinki z jednego zdjęcia nie trafiły do różnych foldów.
    """
    images, labels, groups = [], [], []
    crops_dir = Path(crops_dir)
    for split in splits:
        split_dir = crops_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for crop_path in sorted(class_dir.iterdir()):
                if crop_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = read_image(crop_path)
                if image is None:
                    continue
                images.append(image)
                labels.append(class_dir.name)
                groups.append(source_of_crop(crop_path))
    return images, labels, groups
