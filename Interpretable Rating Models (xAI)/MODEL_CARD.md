# 📄 Model Card: Credit Risk Scoring System

## 1. Szczegóły Modelu (Model Details)

* **Nazwa:** Credit Risk Scoring System (SME/Corporate)
* **Wersja:** 1.0.0
* **Data:** Grudzień 2025
* **Typ:** Klasyfikacja Binarna (Binary Classification)
* **Architektura:** Podejście Dual-Model:
  * **Benchmark:** Regresja Logistyczna (L2 Regularization)
  * **Challenger:** XGBoost (Gradient Boosting Decision Trees)
* **Licencja:** Proprietary / Internal Use Only

## 2. Cel i Zastosowanie (Intended Use)

* **Główny cel:** Predykcja prawdopodobieństwa niewypłacalności (PD - Probability of Default) przedsiębiorstw.
* **Zastosowanie biznesowe:**
  * Automatyzacja decyzji kredytowych dla sektora MŚP.
  * Wyznaczanie limitów kredytowych w oparciu o klasy ratingowe.
  * Kalkulacja rezerw (provisions) i wymogów kapitałowych.
* **Użytkownicy:** Analitycy ryzyka, Komitety Kredytowe.

## 3. Dane (Data)

* **Źródło:** Wewnętrzne dane historyczne instytucji finansowej + dane rejestrowe (PKD, forma własności).
* **Wolumen:** 3000 obserwacji (firm).
* **Zmienna celu (Target):** `default` (1 = upadłość/niewypłacalność, 0 = podmiot zdrowy).
* **Niezbalansowanie:** Default Rate $\approx$ 12.97%.
* **Inżynieria Cech:**
  * Wyselekcjonowano **93 zmienne** (finansowe, behawioralne, makro) metodą Ensemble Selection (Pearson + Mutual Info + RFE + Random Forest).
  * Transformacje: Logarytmizacja zmiennych skośnych, Weight of Evidence (WoE) dla zmiennych kategorycznych, Winsoryzacja (3-97 percentyl).
  * Imputacja: Iterative Imputer (MICE).

## 4. Wyniki i Kalibracja (Performance & Calibration)

Modele zostały poddane rygorystycznej walidacji i kalibracji do tendencji centralnej portfela (**Target PD = 4%**).

### A. Skuteczność Dyskryminacyjna (Test Set)

| Metryka              | Regresja Logistyczna (Benchmark) | XGBoost (Challenger) |
|:-------------------- |:-------------------------------- |:-------------------- |
| **ROC AUC**          | 0.745                            | 0.761                |
| **Gini**             | 0.49                             | 0.522                |
| **Recall (klasa 1)** | 69%                              | 65%                  |

### B. Jakość Kalibracji (Post-Calibration)

Modele zostały skalibrowane metodą hybrydową (Platt Scaling + Target Shifting).

| Metryka                     | Przed Kalibracją (XGB) | Po Kalibracji (XGB) |
| --------------------------- | ---------------------- | ------------------- |
| **ECE (Calibration Error)** | 0.3598                 | **0.0901**          |
| **Brier Score**             | 0.2382                 | **0.1144**          |
| **Średnie PD**              | ~50% (biased)          | **4.00%**           |

Powyższe parametry w przypadku Regresji Logistycznej przyjmują bardzo zbliżone wartości do tych w przypadku XGBoosta. 

## 5. Wyjaśnialność (Explainability / XAI)

W celu zapewnienia pełnej transparentności obu modeli wdrożono wielowarstwowy system interpretacji oparty na metodach agnostycznych oraz, gdzie było to możliwe, wynikających z konstrukcji modeli.

Zastosowano następujące techniki:

### Metody Globalne (Global Interpretability)

1. **Permutation Feature Importance** 

2. **SHAP Global Summary (Beeswarm Plots)** 

3. **Partial Dependence Plots (PDP)** 

4. **2D Interaction Plots**

### Metody Lokalne (Local Interpretability)

1. **SHAP Waterfall**

2. **Individual Conditional Expectation (ICE)**.

3. **LIME (Local Interpretable Model-agnostic Explanations)**

### Global Feature Importance (Top 5 Drivers)

1. **Kapitał Zapasowy (-log):** Najsilniejszy stabilizator; wysoki poziom drastycznie obniża ryzyko (efekt plateau).
2. **Wskaźnik Zadłużenia Ogólnego (log):** Wykładniczy wzrost ryzyka po przekroczeniu bezpiecznego progu (threshold effect).
3. **Struktura Kapitału (Equity to Assets):** Nieliniowa zależność wykryta przez XGBoost.
4. **Efektywność (EBITDA margins):** Kluczowy wskaźnik płynności operacyjnej.
5. **Dźwignia Finansowa:** Szczególnie istotna w modelu liniowym.

### Spójność Modeli

Korelacja rankingowa ważności cech między LR a XGB wynosi jedynie **0.32** (Spearman). Oznacza to, że XGBoost identyfikuje inne, nieliniowe wzorce ryzyka, co uzasadnia jego wdrożenie jako modelu Challenger.

## 6. Ograniczenia i Ryzyka (Limitations & Risks)

* **Mała próba walidacyjna:** Na zbiorze `Unseen Data` (N=300) model Regresji Logistycznej okazał się stabilniejszy (AUC 0.74) niż XGBoost (AUC 0.71). Wskazuje to na ryzyko zmienności modelu złożonego na małych podpróbkach.
* **Horyzont czasowy:** Model jest kalibrowany na specyficzne warunki makroekonomiczne. Zmiana cyklu koniunkturalnego może wymagać rekalibracji (Intercept Shift).

## 7. Plan Monitoringu (Monitoring Plan)

Aby zapewnić bezpieczeństwo wdrożenia, zaleca się kwartalny przegląd modelu obejmujący:

1. **Analiza Stabilności Populacji (PSI):** Monitorowanie zmian w rozkładach zmiennych wejściowych (alert przy PSI > 0.1).
2. **Stabilność Score'u (CSI):** Weryfikacja przesunięć w rozkładzie ratingów.
3. **Backtesting Kalibracji:** Porównanie *Predicted PD* vs *Observed Default Rate* (test Binomialny).
4. **Dyskryminacja:** Sprawdzenie, czy Gini nie spada poniżej 0.60 na nowych rocznikach danych.


