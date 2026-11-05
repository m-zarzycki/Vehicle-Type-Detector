# Vehicle Type Detector

Klasyfikacja zdjęć pojazdów do 5 typów (osobowy, ciężarowy, autobus, motocykl, ambulans) z użyciem cech HOG i klasycznego uczenia maszynowego. Projekt na laboratorium ze sztucznej inteligencji, Collegium Da Vinci, rok akademicki 2026/2027. Założenia opisuje [karta tematu](karta-tematu.md).

## Instalacja

Wymagany jest Python 3.13.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dane

Projekt korzysta ze zbioru [Vehicles-OpenImages](https://universe.roboflow.com/roboflow-gw7yv/vehicles-openimages/dataset/1) (wersja 1, 416×416), autor: Jacob Solawetz, licencja [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Zbiór nie jest przechowywany w repozytorium.

1. Pobierz wersję 1 zbioru w formacie YOLOv8 jako ZIP.
2. Rozpakuj go do `data/raw/`, tak aby powstały foldery `train/`, `valid/`, `test/` i plik `data.yaml`.

## Struktura

```
data/raw/     rozpakowany zbiór z Roboflow (poza gitem)
data/crops/   wycięte pojazdy (poza gitem)
models/       zapisane modele (poza gitem)
notebooks/    notatniki Jupyter
src/          wspólny kod
```
