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
from src.features import HogTransformer, MobileNetTransformer, PixelTransformer

N_FOLDS = 5
RANDOM_STATE = 42
TARGET_ACCURACY = 0.80
TARGET_MACRO_F1 = 0.75
TARGET_RECALL = 0.70


def build_hog_model():
    # C i gamma dobrane walidacją krzyżową na train + valid.
    # gamma="auto" to 1 / liczba cech, a class_weight="balanced" daje rzadszym klasom większą wagę.
    return make_pipeline(
        HogTransformer(),
        StandardScaler(),
        SVC(C=1, gamma="auto", class_weight="balanced"),
    )


def build_mobilenet_model(C=0.3):
    # C do dobrania walidacją krzyżową (np. 0.1, 1, 10), tak jak przy HOG
    return make_pipeline(
        MobileNetTransformer(),
        StandardScaler(),
        SVC(C=C, gamma="auto", class_weight="balanced"),
    )


# Kandydaci na ostateczny model, porównywani w tej samej walidacji krzyżowej
CANDIDATES = {
    "HOG + SVM": build_hog_model,
    "MobileNet + SVM": build_mobilenet_model,
}


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


def meets_targets(accuracy, macro_f1, recalls):
    return accuracy >= TARGET_ACCURACY and macro_f1 >= TARGET_MACRO_F1 and recalls.min() >= TARGET_RECALL


def train(crops_dir=CROPS_DIR, model_path=MODEL_PATH):
    images, labels, groups = load_crops(crops_dir, ("train", "valid"))
    if not images:
        raise FileNotFoundError(f"Brak wycinków w {crops_dir}. Najpierw uruchom: python -m src.make_crops")
    labels = np.array(labels)
    classes = sorted(set(labels))
    print(f"Walidacja krzyżowa ({N_FOLDS} foldów) na {len(images)} wycinkach z train + valid\n")

    models = build_baselines()
    models.update({name: build() for name, build in CANDIDATES.items()})
    results = {name: cross_validate(model, images, labels, groups) for name, model in models.items()}
    for name, predicted in results.items():
        accuracy, macro_f1, recalls = scores(labels, predicted, classes)
        per_class = "  ".join(f"{c} {r:.2f}" for c, r in zip(classes, recalls))
        targets = "tak" if meets_targets(accuracy, macro_f1, recalls) else "nie"
        print(f"{name:<16} accuracy {accuracy:.3f}  macro F1 {macro_f1:.3f}  | recall: {per_class}  | progi: {targets}")

    # Wybór po macro F1 z walidacji krzyżowej, zbiór testowy zostaje nietknięty
    best = max(CANDIDATES, key=lambda name: scores(labels, results[name], classes)[1])
    print(f"\nNajlepszy model: {best}")
    print(classification_report(labels, results[best], digits=3, zero_division=0))

    train_images, train_labels = with_flips(images, labels)
    model = CANDIDATES[best]().fit(train_images, train_labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Zapisano model {best}: {model_path}")
    return labels, results


if __name__ == "__main__":
    train()
