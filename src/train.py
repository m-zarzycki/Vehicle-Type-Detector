import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
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
    # C dobrane walidacją krzyżową na train + valid (python -m src.tune)
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


def features_with_flips(extractor, images):
    """Cechy wycinków i ich odbić lustrzanych.

    Pierwszy krok każdego potoku (piksele, HOG, MobileNet) niczego nie uczy się z danych,
    więc cechy wycinka są takie same w każdym foldzie i wystarczy policzyć je raz.
    """
    features = extractor.transform(images)
    flipped = extractor.transform([cv2.flip(image, 1) for image in images])
    return features, flipped


def cross_validate_features(model, features, flipped, labels, groups):
    """Walidacja krzyżowa na gotowych cechach. Odbicia trafiają tylko do części uczącej foldu."""
    # Tu trafiają predykcje dla wycinków z kolejnych foldów walidacyjnych
    predicted = np.empty_like(labels)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_index, test_index in folds.split(features, labels, groups):
        train_features = np.concatenate([features[train_index], flipped[train_index]])
        train_labels = np.concatenate([labels[train_index], labels[train_index]])
        fold_model = clone(model).fit(train_features, train_labels)
        predicted[test_index] = fold_model.predict(features[test_index])
    return predicted


def cross_validate(model, images, labels, groups):
    """Walidacja krzyżowa potoku: cechy liczone raz pierwszym krokiem, w foldach uczy się reszta potoku."""
    features, flipped = features_with_flips(model[0], images)
    return cross_validate_features(model[1:], features, flipped, labels, groups)


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
    features = {name: features_with_flips(model[0], images) for name, model in models.items()}
    results = {
        name: cross_validate_features(model[1:], *features[name], labels, groups)
        for name, model in models.items()
    }
    for name, predicted in results.items():
        accuracy, macro_f1, recalls = scores(labels, predicted, classes)
        per_class = "  ".join(f"{c} {r:.2f}" for c, r in zip(classes, recalls))
        targets = "tak" if meets_targets(accuracy, macro_f1, recalls) else "nie"
        print(f"{name:<16} accuracy {accuracy:.3f}  macro F1 {macro_f1:.3f}  | recall: {per_class}  | progi: {targets}")

    # Wybór po macro F1 z walidacji krzyżowej, zbiór testowy zostaje nietknięty
    best = max(CANDIDATES, key=lambda name: scores(labels, results[name], classes)[1])
    print(f"\nNajlepszy model: {best}")
    print(classification_report(labels, results[best], digits=3, zero_division=0))

    # Cechy najlepszego modelu są już policzone, więc na całym train + valid uczymy tylko
    # StandardScaler + SVC, a potem doklejamy przed nie krok liczący cechy
    extractor_step, *classifier_steps = CANDIDATES[best]().steps
    classifier = Pipeline(classifier_steps).fit(np.concatenate(features[best]), np.concatenate([labels, labels]))
    model = Pipeline([extractor_step, *classifier.steps])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Zapisano model {best}: {model_path}")
    return labels, results


if __name__ == "__main__":
    train()
