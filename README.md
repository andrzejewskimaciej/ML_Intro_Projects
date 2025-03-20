# ML_Intro-Projects

Celem biznesowym modelu jest zapobieganie oszustwom w transakcjach internetowych poprzez skuteczne wykrywanie fałszywych transakcji. Model uczenia maszynowego ma za zadanie automatycznie identyfikować podejrzane transakcje na podstawie dostępnych cech, takich jak data i godzina zakupu, metoda płatności czy długość istnienia konta w serwisie. Powinien on poprawnie identyfikować jak najwięcej oszustw, nawet kosztem fałszywych wykryć oszustw, aby jak najbardziej zabezpieczyć kupujących przed utratą pieniędzy.

Model tworzony jest przede wszystkim dla właścicieli internetowych platform sprzedażowych, którzy dysponując wszystkimi niezbędnymi danymi o transkacji są w stanie na wczesnym etapie zablokować próby oszustw. Oczywiście musi się to wiązać ze współpracą z bankami, aby po odnotowanej próbie oszustwa na konkretnej witrynie bank całkowicie zablokował możliwość posługiwania się danym środkiem płatniczym przypisanym do konta, z którego próbowano dokonać oszustwa.

Jako główną metrykę efektywności stosować będziemy **Recall**, tj. stosunek wykrytych przypadków oszust do sumy wszystkich oszustw. Dzięki temu model skupiać się będzie na maksymalizacji skuteczności wykrywania oszustw. Będziemy również monitorować **F1 Score** oraz **ROC AUC** w celu ogólnej oceny funkcjonowania modelu.



 

