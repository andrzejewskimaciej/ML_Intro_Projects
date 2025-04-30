## 1. kamień milowy

Eksploracyjna analiza danych została przeprowadzona w sposób poprawny

## 2. kamień milowy

Uwagi do dotychczasowej pracy:

* Czy w augmentacji 1. nie lepiej byłoby (np. przy pomocy analizy jasności obszarów) usuwać wszystkie wykryte fragmenty zawierające nie-chmury (lasy, drogi...), zamiast z p-stwem 0.2 usuwać fragment obrazka lub nie?
  Nawet w przypadku bardzo ciemnego nieba powinien być zauważalny przez algorytm kontrast między obiektami z gruntu, a niebem 
* Może w augmentacji by spróbować stworzyć nowe zdjęcia poprzez dodanie szumu do kanałów kolorów w już istniejących obrazach?
* Fajny pomysł z odcięciem ostatniego layera i dodaniem sklearnowych rzeczy 
* Ogólnie nie wiemy dokładnie co tam się wyczynia, ale wyniki są dobre, wygląda w porządku

## 3. kamień milowy

Model EfficientNet został pomyślnie dotrenowany – osiągnięto satysfakcjonującą dokładność na poziomie 90%. Wysokie wartości recall oraz F1-score wskazują, że model skutecznie rozróżnia poszczególne typy chmur. Dodatkowo zastosowanie metody GradCAM umożliwiło lepsze zrozumienie działania modelu i identyfikację kluczowych obszarów wpływających na podejmowane decyzje. Rekomendujemy pogłębioną analizę wyników GradCAM, ponieważ może ona dostarczyć cennych wskazówek do dalszej optymalizacji modelu i zwiększenia jego interpretowalności. Można również rozważyć dotrenowanie modelu na połączonych zbiorach treningowym i testowym, z zachowaniem tych samych parametrów oraz liczby epok, w celu pełniejszego wykorzystania dostępnych danych.
