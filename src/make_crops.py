import shutil
from collections import Counter
from pathlib import Path

import cv2

from src.config import CROPS_DIR, MIN_BOX_SIDE, RAW_DIR, SPLITS
from src.dataset import YoloDataset, crop_name, read_image, source_name


def write_image(path, image):
    # cv2.imencode + tofile zamiast cv2.imwrite, bo imwrite na Windows nie zapisuje ścieżek z polskimi znakami
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Nie udało się zakodować obrazu: {path}")
    buffer.tofile(str(path))


def make_crops(raw_dir=RAW_DIR, crops_dir=CROPS_DIR, splits=SPLITS, min_side=MIN_BOX_SIDE):
    """Wycina pojazdy z ramek YOLO i zapisuje je do <crops_dir>/<split>/<klasa>/.

    Ramki o krótszym boku poniżej min_side pikseli są odrzucane, bo HOG
    na tak małych wycinkach daje głównie szum. Zwraca liczniki wycinków
    {(split, klasa): liczba}.
    """
    dataset = YoloDataset(raw_dir)
    crops_dir = Path(crops_dir)
    # Stare wycinki usuwamy, żeby po zmianie ustawień nie zostały nieaktualne pliki
    for split in splits:
        shutil.rmtree(crops_dir / split, ignore_errors=True)

    saved = Counter()
    too_small = Counter()
    unreadable = []
    for split, image_path, boxes in dataset.images(splits):
        if not boxes:
            continue
        image = read_image(image_path)
        if image is None:
            unreadable.append(image_path)
            continue
        height, width = image.shape[:2]
        source = source_name(image_path)
        for index, box in enumerate(boxes):
            class_name = dataset.class_name(box.class_id)
            if box.short_side(width, height) < min_side:
                too_small[split, class_name] += 1
                continue
            class_dir = crops_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            write_image(class_dir / crop_name(source, index), box.crop(image))
            saved[split, class_name] += 1

    print_summary(dataset.class_names, splits, saved, too_small, min_side)
    print(f"Pominięte duplikaty zdjęć: {len(dataset.skipped_duplicates)}")
    for split, image_path in dataset.skipped_duplicates:
        print(f"  {split}/{image_path.name}")
    if unreadable:
        print(f"Nieczytelne zdjęcia: {len(unreadable)}")
        for image_path in unreadable:
            print(f"  {image_path}")
    print(f"Zapisano {sum(saved.values())} wycinków w {crops_dir}")
    return saved


def print_summary(class_names, splits, saved, too_small, min_side):
    width = max(len(name) for name in class_names)
    print(f"Wycinki (w nawiasie odrzucone ramki o boku < {min_side} px):")
    print(f"{'':<{width}}  " + "  ".join(f"{split:>12}" for split in splits))
    for name in class_names:
        cells = [f"{saved[split, name]:>5} ({too_small[split, name]:>4})" for split in splits]
        print(f"{name:<{width}}  " + "  ".join(f"{cell:>12}" for cell in cells))


if __name__ == "__main__":
    make_crops()
