"""Dobór C dla MobileNet + SVM walidacją krzyżową na train + valid. Zbiór testowy nie jest używany.

Cechy MobileNet liczą się raz (features_with_flips), a dla każdego C uczy się tylko
StandardScaler + SVM w tych samych foldach co w src/train.py.
"""
import time

import numpy as np

from src.config import CROPS_DIR
from src.dataset import load_crops
from src.train import build_mobilenet_model, cross_validate_features, features_with_flips, meets_targets, scores

C_VALUES = (0.1, 0.3, 1, 3, 10, 30)


def tune(c_values=C_VALUES):
    images, labels, groups = load_crops(CROPS_DIR, ("train", "valid"))
    labels = np.array(labels)
    classes = sorted(set(labels))
    print(f"MobileNet + SVM, walidacja krzyżowa na {len(images)} wycinkach z train + valid\n")

    start = time.perf_counter()
    # Pierwszy krok potoku to MobileNetTransformer, reszta to StandardScaler + SVC
    features, flipped = features_with_flips(build_mobilenet_model()[0], images)
    print(f"Cechy MobileNet policzone raz: {time.perf_counter() - start:.0f} s\n")

    for C in c_values:
        predicted = cross_validate_features(build_mobilenet_model(C)[1:], features, flipped, labels, groups)
        accuracy, macro_f1, recalls = scores(labels, predicted, classes)
        per_class = "  ".join(f"{c} {r:.2f}" for c, r in zip(classes, recalls))
        targets = "tak" if meets_targets(accuracy, macro_f1, recalls) else "nie"
        print(f"C={C:<5} accuracy {accuracy:.3f}  macro F1 {macro_f1:.3f}  | recall: {per_class}  | progi: {targets}")
    print(f"\nCałość: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    tune()
