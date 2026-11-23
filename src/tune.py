"""Dobór C dla MobileNet + SVM walidacją krzyżową na train + valid. Zbiór testowy nie jest używany."""
import numpy as np

from src.config import CROPS_DIR
from src.dataset import load_crops
from src.train import build_mobilenet_model, cross_validate, meets_targets, scores

C_VALUES = (0.1, 0.3, 1, 3, 10, 30)


def tune(c_values=C_VALUES):
    images, labels, groups = load_crops(CROPS_DIR, ("train", "valid"))
    labels = np.array(labels)
    classes = sorted(set(labels))
    print(f"MobileNet + SVM, walidacja krzyżowa na {len(images)} wycinkach z train + valid\n")
    for C in c_values:
        predicted = cross_validate(build_mobilenet_model(C), images, labels, groups)
        accuracy, macro_f1, recalls = scores(labels, predicted, classes)
        per_class = "  ".join(f"{c} {r:.2f}" for c, r in zip(classes, recalls))
        targets = "tak" if meets_targets(accuracy, macro_f1, recalls) else "nie"
        print(f"C={C:<5} accuracy {accuracy:.3f}  macro F1 {macro_f1:.3f}  | recall: {per_class}  | progi: {targets}")


if __name__ == "__main__":
    tune()