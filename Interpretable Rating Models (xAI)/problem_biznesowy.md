## Cel Główny

**Opracowanie modelu ratingowego do oceny zdolności kredytowej i wiarygodności finansowej przedsiębiorstw**, który umożliwi klasyfikację firm według poziomu ryzyka kredytowego i wsparcie decyzji biznesowych.

## Cele Szczegółowe

**1. Predykcja ryzyka niewypłacalności**

- Identyfikacja przedsiębiorstw zagrożonych upadłością lub trudnościami finansowymi
- Wczesne ostrzeganie przed potencjalnymi problemami płatniczymi

**2. Wsparcie procesu decyzyjnego**

- Pomoc w decyzjach kredytowych (udzielanie kredytów, limity, warunki)
- Optymalizacja polityki należności i relacji z kontrahentami
- Wsparcie decyzji inwestycyjnych i partnerskich

**3. Segmentacja portfela klientów**

- Kategoryzacja firm według klas ryzyka (np. AAA, AA, A, BBB, BB, B, CCC, CC, C)
- Umożliwienie zróżnicowanego zarządzania relacjami biznesowymi
- Optymalizacja alokacji zasobów i uwagi analityków

**4. Monitoring i zarządzanie ryzykiem**

- Bieżące monitorowanie kondycji finansowej kontrahentów
- Automatyzacja procesu oceny ryzyka portfela
- Wczesna detekcja pogorszenia sytuacji finansowej

## Korzyści Biznesowe

- **Redukcja strat kredytowych** poprzez lepszą selekcję kontrahentów
- **Automatyzacja i standaryzacja** procesu oceny ryzyka
- **Obiektywizacja decyzji** w oparciu o dane ilościowe
- **Optymalizacja kapitału** pod kątem ryzyka

## Wybór modelu i metryki

Przy opracowywaniu modelu scoringowego kluczowe jest **równoważenie dwóch potrzeb biznesowych**:

1. **Wykrywanie przypadków niewypłacalności (defaultów)** – aby żaden ryzykowny kontrahent nie pozostał niezidentyfikowany.

2. **Unikanie nadmiernego alarmowania** – aby nie klasyfikować nadmiernie wielu przedsiębiorstw jako ryzykowne, co mogłoby prowadzić do utraty potencjalnych klientów lub nadmiernej ostrożności biznesowej.

W związku z tym, jako główną metrykę oceny modelu planuje się użycie **F-beta score**, które pozwala w elastyczny sposób nadawać większą wagę precyzji lub czułości w zależności od priorytetu biznesowego. Rozważane wartości parametru beta to:

- **β = 0.5** – większy nacisk na precyzję (mniej fałszywych alarmów)

- **β = 1** – zrównoważona waga precyzji i czułości

- **β = 2** – większy nacisk na czułość (wykrywanie wszystkich defaultów)

Takie podejście pozwala dobrać model do faktycznych potrzeb biznesowych i strategii zarządzania ryzykiem.
