# 🏦 System Oceny Ryzyka Kredytowego (Credit Risk Scoring)

Ten projekt zawiera kompletny system modelowania ryzyka kredytowego dla przedsiębiorstw. Rozwiązanie opiera się na podejściu **Dual-Model** (Benchmark vs Challenger), integrując tradycyjną Regresję Logistyczną z zaawansowanym modelem XGBoost.

System obejmuje pełen potok przetwarzania: od inżynierii cech, przez selekcję zmiennych i trening modeli, aż po kalibrację probabilistyczną do poziomu 4% i moduł wyjaśnialności (XAI).

Pełny raport techniczny dostępny jest [tutaj](https://github.com/andrzejewskimaciej/ML_Projects/blob/main/Interpretable%20Rating%20Models%20(xAI)/Raport_techniczny.pdf).

---

## 🚀 Kluczowe Funkcjonalności

* **Modele:** Regresja Logistyczna (White-box) oraz XGBoost (Black-box).
* **Kalibracja:** Dostrojenie prawdopodobieństwa niewypłacalności (PD) do tendencji centralnej portfela (4%) metodą Platt Scaling + Target Shifting.
* **Interpretowalność:** Pełna analiza SHAP (Global/Local), wykresy PDP/ICE oraz LIME dla wyjaśniania pojedynczych decyzji.
* **Wdrożenie:** Gotowy moduł "One-Click" do natychmiastowego scoringu nowych zbiorów danych.

---

## 🛠️ Wymagania i Instalacja

Projekt korzysta ze środowiska Conda i wymaga utworzenia środowiska z pliku `environment.yml`.

## 1. Utworzenie środowiska Conda

`conda env create -f environment.yml conda activate <nazwa_środowiska>`

> Nazwa środowiska znajduje się wewnątrz pliku `environment.yml` w sekcji `name:`.

## 2. Uruchomienie notebooka

Notebook testowy znajduje się w pliku:

`ONE-CLICK-TEST.ipynb`

Upewnij się, że uruchamiasz go **z poziomu katalogu projektu**, aby mógł poprawnie wczytać wszystkie potrzebne pliki.

Notebook odpalasz np. komendą:

`jupyter notebook ONE-CLICK-TEST.ipynb`

## 3. Podstawienie własnej ścieżki do danych

W notebooku znajduje się komórka, w której należy podać ścieżkę do własnych danych, np.:

`DATA_PATH = "/ścieżka/do/twoich/danych/"`

Ustaw odpowiednią lokalizację przed uruchomieniem dalszych komórek.
