import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CROPS_DIR, MODEL_PATH
from src.dataset import load_crops
from src.features import HogTransformer, PixelTransformer

N_FOLDS = 5
RANDOM_STATE = 42
TARGET_ACCURACY = 0.80
TARGET_MACRO_F1 = 0.75
TARGET_RECALL = 0.70


def build_model():
    # C i gamma dobrane walidacją krzyżową na train + valid.
    # gamma="auto" to 1 / liczba cech, a class_weight="balanced" daje rzadszym klasom większą wagę.
    return make_pipeline(
        HogTransformer(),
        StandardScaler(),
        SVC(C=1, gamma="auto", class_weight="balanced"),
    )


def build_baselines():
    return {
        "Dummy": make_pipeline(PixelTransformer(), DummyClassifier(strategy="most_frequent")),
        "k-NN piksele": make_pipeline(PixelTransformer(), StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def with_flips(images, labels):
    flipped = [cv2.flip(image, 1) for image in images]
    return images + flipped, np.concatenate([labels, labels])


def cross_validate(model, images, labels, groups):
    # Tu trafiają predykcje dla wycinków z kolejnych foldów walidacyjnych
    predicted = np.empty_like(labels)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_index, test_index in folds.split(images, labels, groups):
        train_images, train_labels = with_flips([images[i] for i in train_index], labels[train_index])
        fold_model = clone(model).fit(train_images, train_labels)
        predicted[test_index] = fold_model.predict([images[i] for i in test_index])
    return predicted


def scores(labels, predicted, classes):
    return (
        accuracy_score(labels, predicted),
        f1_score(labels, predicted, average="macro", zero_division=0),
        recall_score(labels, predicted, labels=classes, average=None, zero_division=0),
    )


def train(crops_dir=CROPS_DIR, model_path=MODEL_PATH):
    images, labels, groups = load_crops(crops_dir, ("train", "valid"))
    if not images:
        raise FileNotFoundError(f"Brak wycinków w {crops_dir}. Najpierw uruchom: python -m src.make_crops")
    labels = np.array(labels)
    classes = sorted(set(labels))
    print(f"Walidacja krzyżowa ({N_FOLDS} foldów) na {len(images)} wycinkach z train + valid\n")

    models = build_baselines()
    models["HOG + SVM"] = build_model()
    results = {name: cross_validate(model, images, labels, groups) for name, model in models.items()}
    for name, predicted in results.items():
        accuracy, macro_f1, recalls = scores(labels, predicted, classes)
        per_class = "  ".join(f"{c} {r:.2f}" for c, r in zip(classes, recalls))
        print(f"{name:<13} accuracy {accuracy:.3f}  macro F1 {macro_f1:.3f}  | recall: {per_class}")

    print("\nHOG + SVM:")
    print(classification_report(labels, results["HOG + SVM"], digits=3, zero_division=0))
    accuracy, macro_f1, recalls = scores(labels, results["HOG + SVM"], classes)
    meets_targets = (
        accuracy >= TARGET_ACCURACY
        and macro_f1 >= TARGET_MACRO_F1
        and recalls.min() >= TARGET_RECALL
    )
    print("Progi z karty spełnione:", "tak" if meets_targets else "nie")

    train_images, train_labels = with_flips(images, labels)
    model = build_model().fit(train_images, train_labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Zapisano model: {model_path}")


if __name__ == "__main__":
    train()