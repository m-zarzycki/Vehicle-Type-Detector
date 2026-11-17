import argparse

from src.classifier import VehicleClassifier
from src.config import MODEL_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Rozpoznaje typ pojazdu na zdjęciu. Najlepiej działa, gdy pojazd wypełnia większość kadru."
    )
    parser.add_argument("images", nargs="+", help="ścieżki do zdjęć")
    parser.add_argument("--model", default=MODEL_PATH, help="plik modelu")
    args = parser.parse_args()

    classifier = VehicleClassifier(args.model)
    for image_path in args.images:
        try:
            print(f"{image_path}: {classifier.predict(image_path)}")
        except (FileNotFoundError, ValueError) as error:
            print(f"{image_path}: błąd - {error}")


if __name__ == "__main__":
    main()