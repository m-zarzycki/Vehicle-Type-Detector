# Vehicle Type Detector

Klasyfikacja zdjęć pojazdów do 5 typów (osobowy, ciężarowy, autobus, motocykl, ambulans) klasyfikatorem SVM. Porównujemy dwa rodzaje cech: HOG oraz cechy z gotowej sieci MobileNetV2, którą przewiduje plan awaryjny z karty. Projekt na laboratorium ze sztucznej inteligencji, Collegium Da Vinci, rok akademicki 2026/2027. Założenia opisuje [karta tematu](karta-tematu.md).

## Instalacja

Wymagany jest Python 3.13.

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` zawiera PyTorch (`torch`, `torchvision`), więc instalacja pobiera kilkaset MB. Wagi MobileNetV2 (około 14 MB) pobierają się automatycznie przy pierwszym uruchomieniu treningu do `~/.cache/torch`.

## Dane

Projekt korzysta ze zbioru [Vehicles-OpenImages](https://universe.roboflow.com/roboflow-gw7yv/vehicles-openimages/dataset/1) (wersja 1, 416×416), autor: Jacob Solawetz, licencja [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Zbiór nie jest przechowywany w repozytorium.

1. Pobierz wersję 1 zbioru w formacie YOLOv8 jako ZIP.
2. Rozpakuj go do `data/raw/`, tak aby powstały foldery `train/`, `valid/`, `test/` i plik `data.yaml`. Jeśli po rozpakowaniu znajdą się one w podfolderze (np. `data/raw/Vehicles-OpenImages.v1-416x416.yolov8/`), przenieś jego zawartość bezpośrednio do `data/raw/`.

Zbiór ma 1254 zdjęcia (train 878, valid 250, test 126) z ramkami w formacie YOLO dla 5 klas: Ambulance, Bus, Car, Motorcycle, Truck.

### Duplikaty

Każde zdjęcie występuje w zbiorze **dokładnie dwa razy** pod różnymi nazwami nadanymi przez Roboflow, np. `005980b3a5f93085_jpg.rf.2QZC….jpg` i `005980b3a5f93085_jpg.rf.68f1….jpg`. Obie kopie mają identyczne piksele i etykiety i zawsze leżą w tym samym podziale, więc między train, valid i test nie ma przecieku.

Gdyby duplikaty zostały w danych, kopia tego samego zdjęcia mogłaby trafić do innego foldu walidacji krzyżowej niż oryginał i zawyżyć wynik. `YoloDataset` (`src/dataset.py`) rozpoznaje duplikaty po nazwie przed `.rf.` i zostawia tylko pierwsze wystąpienie. Po ich usunięciu zostaje **627 unikalnych zdjęć**: train 439, valid 125, test 63. Listę pominiętych plików wypisuje `python -m src.make_crops`.

### Wycinki pojazdów

`python -m src.make_crops` wycina pojazdy z ramek i odrzuca ramki o krótszym boku poniżej 32 px (`MIN_BOX_SIDE` w `src/config.py`). Z 1194 ramek powstaje **997 wycinków**:

| Klasa      | train | valid | test | razem | odrzucone (< 32 px) |
|------------|------:|------:|-----:|------:|--------------------:|
| Ambulance  |    83 |    32 |    9 |   124 |                   2 |
| Bus        |    89 |    23 |   18 |   130 |                  11 |
| Car        |   341 |    93 |   51 |   485 |                 166 |
| Motorcycle |    94 |    23 |   15 |   132 |                   8 |
| Truck      |    88 |    28 |   10 |   126 |                  10 |
| **Razem**  |   695 |   199 |  103 |   997 |                 197 |

Klasa Car stanowi około 49% wycinków, każda z pozostałych około 13%. Model kompensuje to wagami klas (`class_weight="balanced"`).

Wycinki z train i valid (894) służą do 5-krotnej walidacji krzyżowej (`StratifiedGroupKFold`). Wycinki z jednego zdjęcia zawsze trafiają do tego samego foldu. Wycinki z test (103) są odłożone do końcowej oceny wybranego modelu.

## Modele

Oba modele używają tego samego klasyfikatora (`StandardScaler` + `SVC` z jądrem RBF i `class_weight="balanced"`). Różnią się cechami i parametrem `C`:

- **HOG + SVM** (`build_hog_model` w `src/train.py`): wycinek w skali szarości 128×128 zamieniony na 2352 cechy HOG, czyli kierunki krawędzi.
- **MobileNet + SVM** (`build_mobilenet_model`): wycinek 224×224 przepuszczony przez MobileNetV2 wytrenowaną na ImageNet, bez ostatniej warstwy. Daje to 1280 cech. Sieć nie jest dotrenowywana, uczy się tylko SVM.

`python -m src.train` porównuje oba modele z baseline'ami (Dummy, k-NN na pikselach) w walidacji krzyżowej, wybiera ten z wyższym macro F1, trenuje go na całym train + valid i zapisuje do `models/vehicle_classifier.joblib`.

## Uruchomienie

```bash
python -m src.make_crops          # wycięcie pojazdów do data/crops/
python -m src.train               # porównanie modeli i zapis najlepszego do models/
python -m src.tune                # dobór parametru C dla MobileNet + SVM
python predict.py zdjecie.jpg     # typ pojazdu dla podanych zdjęć
```

`src.train` liczy cechy MobileNet osobno dla każdego foldu, więc na CPU trwa kilka minut. `predict.py` działa najlepiej, gdy pojazd wypełnia większość kadru, bo model uczył się na wycinkach.

Notatniki w `notebooks/`:
- `analiza_bledow_hog_svm.ipynb`: macierz pomyłek i przykłady błędów HOG + SVM, zwłaszcza dla klasy Truck,
- `proba_poprawy_hog_svm.ipynb`: porównanie wariantów: balansowanie klas, augmentacja, kolor, kształt.

## Struktura

```
data/raw/     rozpakowany zbiór z Roboflow (poza gitem)
data/crops/   wycięte pojazdy (poza gitem)
models/       zapisane modele (poza gitem)
notebooks/    notatniki Jupyter
src/          wspólny kod
```
