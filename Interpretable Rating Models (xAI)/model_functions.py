from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    QuantileTransformer,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import VarianceThreshold
from sklearn import set_config

set_config(transform_output="pandas")
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
)
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.selection import DropCorrelatedFeatures, DropConstantFeatures
from feature_engine.encoding import WoEEncoder
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import pickle


# które kolumny nie powinny zawierać braków danych
zero_as_missing = {
    "szczegolnaFormaPrawna_Symbol": False,
    "formaWlasnosci_Symbol": False,
    "pkdKod": False,
    "wsk_liczba_dni_istnienia": False,
    "Aktywa": True,
    "Aktywa_trwale": True,
    "Wartosci_niematerialne_prawne": True,
    "Wartosc_firmy": True,
    "Rzeczowe_aktywa_trwale": True,
    "Srodki_trwale": True,
    "Naleznosci_dlugoterminowe": True,
    "Inwestycje_dlugoterminowe": True,
    "Rozliczenia_miedzyokresowe_dlugie": True,
    "Aktywa_obrotowe": True,
    "Zapasy": False,
    "Naleznosci_krotkoterminowe": True,
    "Naleznosci_dostaw_uslug_12m_powiazane": True,
    "Naleznosci_dostaw_uslug_pow12m_powiazane": True,
    "Naleznosci_dostaw_uslug_12m_kapitale": True,
    "Naleznosci_dostaw_uslug_pow12m_kapitale": True,
    "Naleznosci_dostaw_uslug_12m_pozostale": True,
    "Naleznosci_dostaw_uslug_pow12m_pozostale": True,
    "Naleznosci_dostaw_uslug_pozostale_sadowe": True,
    "Inwestycje_krotkoterminowe": False,
    "Srodki_pieniezne": False,
    "Rozliczenia_miedzyokresowe_krotkie": False,
    "Kapital_wlasny": True,
    "Kapital_podstawowy": True,
    "Kapital_zapasowy": True,
    "Zysk_netto": False,
    "Zobowiazania_rezerwy": False,
    "Rezerwy_na_zobowiazania": False,
    "Rezerwa_z_tytulu_odroczonego_podatku_dochodowego": False,
    "Rezerwa_na_swiadczenia_emerytalne": False,
    "Rezerwa_na_swiadczenia_emerytalne_dlugie": False,
    "Rezerwa_na_swiadczenia_emerytalne_krotkie": False,
    "Pozostale_rezerwy": False,
    "Pozostale_rezerwy_krotkie": False,
    "Zobowiazania_dlugoterminowe": False,
    "Kredyty_pozyczki_dlugie": False,
    "Inne_zobowiazania_fin_dlugoterminowe": False,
    "Zobowiazania_krotkoterminowe": False,
    "Zobowiazania_dostaw_uslug_12m_powiazane": False,
    "Zobowiazania_dostaw_uslug_pow12m_powiazane": False,
    "Zobowiazania_dostaw_uslug_12m_kapitale": False,
    "Zobowiazania_dostaw_uslug_pow12m_kapitale": False,
    "Kredyty_pozyczki_krotkie": False,
    "Inne_zobowiazania_fin_krotkoterminowe": False,
    "Zobowiazania_dostaw_uslug_12m_pozostale": False,
    "Zobowiazania_dostaw_uslug_pow12m_pozostale": False,
    "Rozliczenia_miedzyokresowe_b": False,
    "Ujemna_wartosc_firmy": True,
    "Inne_rozliczenia_miedzyokresowe": False,
    "Inne_rozliczenia_miedzyokresowe_dlugie": False,
    "Inne_rozliczenia_miedzyokresowe_krotkie": False,
    "schemat_wsk_bilans": False,
    "Naleznosci_dostaw_uslug_12m": True,
    "Naleznosci_dostaw_uslug_pow12m": True,
    "Zobowiazania_dostaw_uslug_12m": True,
    "Zobowiazania_dostaw_uslug_pow12m": True,
    "Kredyty_pozyczki": False,
    "wsk_kapital_do_aktywa": True,
    "przychody_sprzedazy": True,
    "koszty_sprzedanych_produktow": False,
    "koszty_sprzedazy": False,
    "koszty_ogolnego_zarzadu": False,
    "zysk_sprzedazy": False,
    "pozostale_przychody_oper": False,
    "dotacje": False,
    "koszty_operacyjne_pozostale": False,
    "zysk_operacyjny": False,
    "przychody_finansowe": False,
    "dywidendy_udzialy": False,
    "przychody_odsetki": False,
    "koszty_finansowe": False,
    "koszty_odsetki": False,
    "zysk_brutto": False,
    "podatek_dochodowy": False,
    "zysk_netto": False,
    "koszty_operacyjne": False,
    "amortyzacja": True,
    "schemat_wsk_rzis": False,
    "przychody": True,
    "wsk_amortyzacja": True,
    "wsk_koszty_operacyjne": True,
    # Wskaźniki finansowe – traktujemy 0 jako brak danych
    "wsk_Zobowiazania_krotkoterminowe": True,
    "wsk_Zobowiazania_dlugoterminowe": True,
    "wsk_marza_brutto_1": True,
    "wsk_marza_brutto_2": True,
    "wsk_stopa_marzy_brutto": True,
    "wsk_rent_operacyjna": True,
    "wsk_ROS": True,
    "wsk_ROA": True,
    "wsk_s_ROA": True,
    "wsk_rent_operacyjna_aktywow": True,
    "wsk_ROE": True,
    "wsk_s_ROE": True,
    "wsk_mnoznik_kap_wl": True,
    "wsk_zwrot_aktywa_trwale": True,
    "wsk_rent_kaptial_podstawowy": True,
    "wsk_akt_generowania_got_1": True,
    "wsk_rent_sprzedazy": True,
    "wsk_ebit": True,
    "wsk_ebitda_1": True,
    "wsk_ebitda_2": True,
    "wsk_ebitda_3": True,
    "wsk_marza_ebitda_1": True,
    "wsk_marza_ebitda_2": True,
    "wsk_marza_ebitda_3": True,
    "wsk_marza_ebit": True,
    "wsk_ebitda_aktywa_1": True,
    "wsk_ebitda_aktywa_2": True,
    "wsk_ebitda_aktywa_3": True,
    "wsk_zwrot_aktywa_mat": True,
    "wsk_zysk_zobowiazania": True,
    "wsk_zysk_op_zobowiazania": True,
    "wsk_sprzedaz_kap_obrotowy": True,
    "wsk_koszty_przychody": True,
    "wsk_rent_kapitalu": True,
    "wsk_stopa_zysku_sprzedaz": True,
    "wsk_pokrycie_wyd_fin_gotowkowe_1": True,
    "wsk_koszt_długu_1": True,
    "wsk_koszt_długu_2": True,
    "wsk_pokrycie_aktywow_tr_kapitalem_st": True,
    "wsk_struktury_finansowania": True,
    "wsk_pokrycie_zob_kr_gotowkowe_1": True,
    "wsk_zysk_operacyjny_zob_1": True,
    "wsk_zysk_operacyjny_zob_2": True,
    "wsk_zadluzenia_gotowki_1": True,
    "wsk_koszty_fin_przychody": True,
    "wsk_koszty_odsetki_przychody": True,
    "wsk_zadluzenie_gotowka": True,
    "wsk_udzial_kap_wlasnego_aktywa_1": True,
    "wsk_udzial_kap_wlasnego_aktywa_2": True,
    "wsk_ogolnego_zadluzenia_1": True,
    "wsk_ogolnego_zadluzenia_2": True,
    "wsk_pokrycie_aktywow_kap_stalym": True,
    "wsk_zadluzenie_kap_wlasnego": True,
    "wsk_ogolnego_zadluzenia_pozyczki": True,
    "wsk_zadluzenia_pozyczki_dlugie": True,
    "wsk_zadluzenia_dlugie": True,
    "wsk_zadluzenia_krotkie": True,
    "wsk_pokrycia_zobowiazan_rz_aktywami_trwalymi": True,
    "wsk_ROE_brutto": True,
    "wsk_ROA_operacyjny": True,
    "wsk_efekt_dzwigni_fin_1": True,
    "wsk_efekt_dzwigni_fin_2": True,
    "wsk_pokrycia_odsetek_zyskiem": True,
    "wsk_ebitda_koszty_odsetkowe_1": True,
    "wsk_ebitda_koszty_odsetkowe_2": True,
    "wsk_ebitda_koszty_odsetkowe_3": True,
    "wsk_ebitda_koszty_finansowe_1": True,
    "wsk_ebitda_koszty_finansowe_2": True,
    "wsk_ebitda_koszty_finansowe_3": True,
    "wsk_ebitda_zobowiazan_odsetki_1": True,
    "wsk_ebitda_zobowiazan_odsetki_2": True,
    "wsk_ebitda_zobowiazan_odsetki_3": True,
    "wsk_ebitda_zobowiazan_odsetki_4": True,
    "wsk_ebitda_zobowiazan_1": True,
    "wsk_ebitda_zobowiazan_2": True,
    "wsk_ebitda_zobowiazan_3": True,
    "wsk_rotacja_aktywow_1": True,
    "wsk_rotacja_aktywow_2": True,
    "wsk_rotacja_rz_aktywow_trwalych": True,
    "wsk_rotacja_wartosci_niewaterialnych": True,
    "wsk_rotacja_zapasow": False,
    "wsk_rotacja_naleznosci": False,
    "wsk_rotacja_naleznosci_dostaw_uslug": False,
    "wsk_cykl_operacyjny": False,
    "wsk_poziom_kosztow_operacyjnych": False,
    "wsk_poziom_kosztow_finansowych": False,
    "wsk_obrotowsci_naleznosci": False,
    "wsk_rotacja_zobowiazan": False,
    "wsk_rotacja_zobowiazan_dostaw_uslug": False,
    "wsk_cykl_konwersji_gotowki": False,
    "wsk_plynnosc_biez_1": True,
    "wsk_plynnosc_biez_2": True,
    "wsk_plynnosc_biez_3": True,
    "wsk_plynnosc_szybka_1": False,
    "wsk_plynnosc_szybka_2": False,
    "wsk_plynnosc_gotowkowa_1": False,
    "wsk_poziom_kapitalu_obrotowego_netto": False,
    "wsk_udzial_kapitalu_obrotowego_netto": False,
    "wsk_udzial_zob_biez_sprzedaz_1": True,
    "wsk_udzial_zob_biez_sprzedaz_2": False,
    "wsk_udzial_zob_biez_aktywa_1": True,
    "wsk_udzial_zob_biez_aktywa_2": True,
    "wsk_udzial_zapasy_zobowiazania": False,
    "wsk_udzial_zapasy_kap_obrotowy": False,
    "wsk_udzial_kap_obrotowego_w_fin": False,
    "wsk_zysk_ebitda_1": True,
    "wsk_zysk_ebitda_2": True,
    "wsk_zysk_ebitda_3": True,
    "wsk_obrotowosc_gotowkowa": False,
    "wsk_struktura_majatku": True,
    "wsk_struktury_kapitalu": True,
    "wsk_zast_kapitalu_wlasnego": True,
    "wsk_zast_kapitalu_podstawowego": True,
    "wsk_zast_kapitalu_stalego": True,
    "wsk_zast_kapitalu_obcego": True,
    "wsk_sytuacji_fin": True,
    "wsk_struktura_kap_wlasnego_1": True,
    "wsk_struktura_kap_wlasnego_2": True,
    "wsk_struktura_kap_wlasnego_s_1": True,
    "wsk_struktura_kap_wlasnego_s_2": True,
    "wsk_zadluzenia": True,
    "wsk_zob_dlugoterminowe_aktywa_rzeczowe": True,
    "wsk_zob_oprocentowanych": True,
    "wsk_zob_oprocentowanych_aktywa_rzeczowe": True,
    "wsk_struktura_kap_obcego_s": True,
    "wsk_zob_s_aktywa_rzeczowe": True,
    "wsk_fin_majatku_kapitalem": True,
    "default": False,
}

# kolumny, które nie powinny zawierać zer
shouldnt_be_0 = pd.Series(zero_as_missing.keys())[
    list(zero_as_missing.values())
].to_list()


def dataFullTransformer(X_cols: pd.Index):
    """Tworzy Pipeline do całościowego przetworzenia danych gotowych do wprowadzenia do modelu"""
    corr_threshold = 0.7

    # kolumny do usunięcia
    columns_to_remove = [
        # kategoryczne, nie wiadomo co to
        "schemat_wsk_bilans",
        "schemat_wsk_rzis",
        # tylko jedna wartość
        "szczegolnaFormaPrawna_Symbol",
    ]

    # kolumny do przetworzenia (numeryczne kody i symbole)
    cols_to_group = ["pkdKod", "formaWlasnosci_Symbol"]

    numeric_cols = [
        col for col in X_cols.to_list() if col not in columns_to_remove + cols_to_group
    ]

    numeric_pipeline = Pipeline(
        [
            # zastąpienie nieskończoności przez np.nan
            (
                "replaceInf",
                FunctionTransformer(replaceInf, feature_names_out="one-to-one"),
            ),
            # tam, gdzie zero jest brakiem danych wstawiamy nan i usuwamy kolumny, w których jest za dużo braków danych ( > 80%)
            (
                "replace0missings",
                ZeroToNaNTransformer(),
            ),
            # usunięcie kolumn bez wartości (nie da się na nich wyliczyć wariancji)
            ("removeEmptyColumns", DropConstantFeatures(missing_values="include")),
            # usunięcie outlierów powyżej 95 centyla i ponieżej 5 centyla
            ("removeOutliers1", OutliersClipper(0.95)),
            # usunięcie kolumn niskiej wariancji
            ("removeLowVariance1", VarianceThreshold(threshold=0.5)),
            # imputacja danych
            (
                "iterativeImputer",
                IterativeImputer(
                    missing_values=np.nan,
                    initial_strategy="median",
                    max_iter=40,
                    n_nearest_features=40,
                    min_value=-1e6,
                    max_value=1e6,
                    random_state=47,
                ),
            ),
            # dodanie dodatkowych cech finansowych
            (
                "financial_engineering",
                FinancialFeatureEngineer(),
            ),
            # clip danych, aby nie było zbyt dużych wartości
            (
                "dataClip",
                FunctionTransformer(
                    clip,
                    feature_names_out="one-to-one",
                ),
            ),
            # usunięcie kolumn niskiej wariancji
            ("removeLowVariance2", VarianceThreshold(threshold=0.5)),
            # standaryzacja
            ("standarization", StandardScaler()),
            # usunięcie skorelowanych kolumn
            ("dropCorrelated", DropCorrelatedFeatures(threshold=corr_threshold)),
            # ustawienie wszystkich korelacji na dodatnich
            ("signCorrelationFlip", SignCorrelationTransformer()),
        ]
    )

    symbols_pipeline = Pipeline(
        [
            # grupowanie pkd kod i formy działalności
            ("symbolsToGroups", symbolsTransformer()),
            # woe na utworzonych grupach
            ("symbolsWOE", WoEEncoder(fill_value=1)),
        ]
    )

    return ColumnTransformer(
        [
            # drop niepotrzebnych i głównie pustych kolumn
            ("dropper", "drop", columns_to_remove),
            # pipeline numeryczny
            ("numeric_pipeline", numeric_pipeline, numeric_cols),
            # pipeline symboli
            ("symbols_pipeline", symbols_pipeline, cols_to_group),
        ]
    )


def processing_predictingPipeline(model_name: str):
    models = {"LR": "LogisticRegression", "XGB": "XGBoost"}
    model_long_name = models[model_name]
    with open(f"models/Calibrated{model_long_name}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"models/dataPreprocessor.pkl", "rb") as f:
        dataTrans = pickle.load(f)
    with open("models/selected_features.pkl", "rb") as f:
        cols_to_remain = pickle.load(f)

    return Pipeline(
        [
            ("dataTransformation", dataTrans),
            ("dropper", FeatureSelector(cols_to_remain)),
            ("model", model),
        ]
    )


def evaluate_model_side_by_side(
    model, name, X_train, y_train, X_test, y_test, threshold=0.5, val=False, vis=True
):
    train_val_text = "Val" if val else "Test"
    """
    Tworzy zestawienie metryk train/test obok siebie oraz rysuje confusion matrix.
    """
    print("#" * 15)
    print(name)
    labels = ["0", "1"]
    # Predykcje
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_train = (y_proba_train > threshold).astype(int)
    y_pred_test = (y_proba_test > threshold).astype(int)

    # --- CZĘŚĆ 1: Metryki per klasa (Precision, Recall, F1) ---

    # Classification reports jako słowniki
    report_train = classification_report(
        y_train, y_pred_train, target_names=labels, output_dict=True
    )
    report_test = classification_report(
        y_test, y_pred_test, target_names=labels, output_dict=True
    )

    # Budujemy dataframe z metrykami obok siebie
    metrics = ["precision", "recall", "f1-score", "support"]
    data = {}
    for m in metrics:
        train_vals = [
            report_train[c][m] if c in report_train else report_train["macro avg"][m]
            for c in report_train
            if c not in ("accuracy", "macro avg", "weighted avg")
        ]
        test_vals = [
            report_test[c][m] if c in report_test else report_test["macro avg"][m]
            for c in report_test
            if c not in ("accuracy", "macro avg", "weighted avg")
        ]
        col_train = [f"{m}_train"] * len(train_vals)
        col_test = [f"{m}_{train_val_text.lower()}"] * len(test_vals)
        data.update(
            {f"{m}_train": train_vals, f"{m}_{train_val_text.lower()}": test_vals}
        )

    index_labels = [
        c for c in report_train if c not in ("accuracy", "macro avg", "weighted avg")
    ]
    df_metrics = pd.DataFrame(data, index=index_labels)
    print(f"\n=== Metryki Train vs {train_val_text} ===")
    display(df_metrics)

    # --- CZĘŚĆ 2: Metryki Globalne (AUC, PR-AUC, KS, LogLoss, Brier) ---

    def calculate_global_metrics(y_true, y_prob):
        # KS Statistic calculation
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks_stat = np.max(tpr - fpr)

        return {
            "ROC AUC": roc_auc_score(y_true, y_prob),
            "PR AUC": average_precision_score(y_true, y_prob),
            "KS Stat": ks_stat,
            "Log Loss": log_loss(y_true, y_prob),
            "Brier Score": brier_score_loss(y_true, y_prob),
        }

    global_train = calculate_global_metrics(y_train, y_proba_train)
    global_test = calculate_global_metrics(y_test, y_proba_test)

    df_global = pd.DataFrame(
        {
            "Metric": global_train.keys(),
            "Train": global_train.values(),
            f"{train_val_text}": global_test.values(),
        }
    ).set_index("Metric")

    df_global.loc["Accuracy", "Train"] = report_train["accuracy"]
    df_global.loc["Accuracy", f"{train_val_text}"] = report_test["accuracy"]

    print(f"\n=== Metryki Globalne Probabilistyczne ===")
    display(df_global.round(4))

    # --- CZĘŚĆ 3: Wizualizacje ---
    if vis:
        # Confusion matrices
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Train Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Oranges", ax=axes[1])
        axes[1].set_title(f"{train_val_text} Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        plt.show()

        # fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
        # auc = roc_auc_score(y_test, y_proba_test)

        # plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        # plt.plot([0, 1], [0, 1], "k--")  # linia losowa
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("ROC curve")
        # plt.legend()
        # plt.show()

        # ROC Curve (Train vs Test)
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_proba_train)
        fpr_te, tpr_te, _ = roc_curve(y_test, y_proba_test)

        auc_tr = global_train["ROC AUC"]
        auc_te = global_test["ROC AUC"]

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_tr, tpr_tr, label=f"Train AUC = {auc_tr:.3f}", color="blue")
        plt.plot(
            fpr_te,
            tpr_te,
            label=f"{train_val_text} AUC = {auc_te:.3f}",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

    return


def evaluate_model_single_set(
    model, name, X, y, threshold=0.5, dataset_name="Test", vis=True
):
    """
    Tworzy zestawienie metryk dla pojedynczego zbioru danych oraz rysuje
    confusion matrix i krzywą ROC.
    """
    print("#" * 30)
    print(f"Ewaluacja modelu: {name}")
    print(f"Zbiór danych: {dataset_name}")
    print(f"Użyty próg (threshold): {threshold}")
    print("#" * 30)

    labels = [0, 1]

    # 1. Predykcje
    # Sprawdzamy czy model ma predict_proba (większość ma)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        # Fallback dla modeli typu SVM bez włączonego probability
        y_proba = model.predict(X)

    # Aplikacja progu
    y_pred = (y_proba > threshold).astype(int)

    # --- CZĘŚĆ 1: Metryki per klasa (Precision, Recall, F1) ---

    # Generowanie raportu
    report = classification_report(y, y_pred, target_names=["0", "1"], output_dict=True)

    # Konwersja do DataFrame w czytelnej formie
    df_metrics = pd.DataFrame(report).transpose()

    # Opcjonalnie: filtrowanie wierszy jak w oryginale (tylko klasy),
    # ale tutaj zostawiam całość, bo przy jednym zbiorze accuracy/macro avg się przydają.
    # Jeśli chcesz tylko klasy 0 i 1:
    # df_metrics = df_metrics.loc[["0", "1"]]

    print(f"\n=== Metryki Klasyfikacji ({dataset_name}) ===")
    display(df_metrics)

    # --- CZĘŚĆ 2: Metryki Globalne (AUC, PR-AUC, KS, LogLoss, Brier) ---

    def calculate_global_metrics(y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks_stat = np.max(tpr - fpr)

        return {
            "ROC AUC": roc_auc_score(y_true, y_prob),
            "PR AUC": average_precision_score(y_true, y_prob),
            "KS Stat": ks_stat,
            "Log Loss": log_loss(y_true, y_prob),
            "Brier Score": brier_score_loss(y_true, y_prob),
        }

    global_metrics = calculate_global_metrics(y, y_proba)

    # Dodajemy Accuracy ręcznie do globalnych
    global_metrics["Accuracy"] = accuracy_score(y, y_pred)

    df_global = pd.DataFrame(
        global_metrics.values(), index=global_metrics.keys(), columns=[dataset_name]
    )

    print(f"\n=== Metryki Globalne Probabilistyczne ===")
    display(df_global.round(4))

    # --- CZĘŚĆ 3: Wizualizacje ---
    if vis:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title(
            f"Confusion Matrix ({dataset_name})\nThreshold={threshold:.3f}"
        )
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_val = global_metrics["ROC AUC"]

        axes[1].plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", color="darkorange", lw=2)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)  # Linia losowa
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(f"ROC Curve ({dataset_name})")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return df_global


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.fitted_ = True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        # Po prostu zwraca wybrane kolumny
        # Zwracamy DataFrame, żeby zachować nazwy kolumn dla modelu
        return X[self.feature_names]

    def get_feature_names_out(self, input_features=None):
        # To naprawia błędy walidacji nazw w sklearn
        return self.feature_names


def drop_all_na(X):
    return X.dropna(axis=1, how="all")


def clip(X):
    return X.clip(lower=-1e15, upper=1e15)


def feature_names_out_drop_all_na(X_in, transformer=None):
    return X_in.columns[X_in.notna().any(axis=0)]


def replaceInf(X):
    if not isinstance(X, pd.DataFrame):
        raise Exception("X is not an instance of pd.DataFrame")
    X_ = X.copy()
    X_ = X_.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    return X_


class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [col for col in input_features if col not in self.columns_to_drop]

    def __init__(self):
        self.columns_to_drop = []
        self.columns_to_replace = []

    def fit(self, X, y=None):
        self.fitted_ = True
        X_ = X.copy()
        self.columns_to_replace = [
            col for col in X_.columns.to_list() if col in shouldnt_be_0
        ]
        self.columns_to_drop = X_.columns[np.where(X_.isna().mean() > 0.8)]
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        X_ = X.copy()
        X_[self.columns_to_replace] = X_[self.columns_to_replace].replace(
            to_replace=0, value=np.nan
        )
        X_ = X_.drop(columns=self.columns_to_drop)

        return X_


class SignCorrelationTransformer(BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        ret = []
        for col in input_features:
            if col in self.columns_to_flip:
                ret.append(f"-{col}")
            else:
                ret.append(col)
        return ret

    def __init__(self):
        self.columns_to_flip = []

    def fit(self, X, y):
        # Zabezpieczenie na różne typy y
        self.fitted_ = True
        if y is None:
            raise ValueError("SignCorrelationTransformer wymaga podania y przy fit().")

        # Zamiana y na 1D np. Series
        if isinstance(y, (pd.DataFrame, np.ndarray)):
            y = np.ravel(y)

        for col in X.columns:
            correlation, _ = pearsonr(X[col], y)
            if correlation < 0:
                self.columns_to_flip.append(col)
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        X_ = X.copy()
        X_[self.columns_to_flip] = X_[self.columns_to_flip] * -1
        return X_


class OutliersClipper(BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        return input_features

    def __init__(self, quantile=0.95):
        self.q_dict = {}
        self.quantile = quantile

    def fit(self, X, y=None):
        self.fitted_ = True
        df = X.copy()
        for col in df.columns.to_list():
            if df[col].max() > 1 or df[col].min() < 0:
                q = df[col].quantile(self.quantile)
                self.q_dict[col] = q
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "fitted_")
        df = X.copy()
        for col, q in self.q_dict.items():
            df[col] = df[col].clip(lower=None, upper=q)
        return df


class symbolsTransformer(BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        return input_features

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "fitted_")
        df = X.copy()

        def kategoria_pkd(kod):
            if kod == 0:
                return "Nieznane / brak danych"
            elif 100 <= kod < 400:  # np. 1011, 142, 236
                return "Rolnictwo, leśnictwo, rybactwo"
            elif 500 <= kod < 1000:
                return "Górnictwo i wydobywanie"
            elif 1000 <= kod < 3500:  # np. 1623, 2562, 3320
                return "Przemysł i produkcja"
            elif 3500 <= kod < 4500:  # np. 3511, 4120, 4110
                return "Energetyka i budownictwo"
            elif 4500 <= kod < 4800:  # np. 4646, 4633, 4671
                return "Handel hurtowy i detaliczny"
            elif 4800 <= kod < 5500:
                return "Transport i magazynowanie"
            elif 5500 <= kod < 7000:
                return "Zakwaterowanie, gastronomia, IT, finanse"
            elif 7000 <= kod < 8000:
                return "Doradztwo, działalność profesjonalna"
            elif 8000 <= kod < 9000:
                return "Administracja, edukacja, zdrowie"
            else:
                return "Inne usługi"

        df["pkdKod"] = df["pkdKod"].astype("object")
        df["pkdKod"] = df["pkdKod"].apply(kategoria_pkd)

        # --- 2️⃣ Grupowanie form własności ---
        def kategoria_wlasnosci(kod):
            if kod in [111, 112, 113, 121, 122, 123, 131, 132, 133]:
                return "Sektor publiczny"
            elif kod in [214, 215, 224, 225]:
                return "Sektor prywatny krajowy"
            elif kod in [216, 226, 236]:
                return "Sektor prywatny zagraniczny"
            elif kod in [234, 235]:
                return "Sektor mieszany krajowy"
            elif kod == 0:
                return "Brak danych"
            else:
                return "Inna forma"

        df["formaWlasnosci_Symbol"] = df["formaWlasnosci_Symbol"].astype("object")
        df["formaWlasnosci_Symbol"] = df["formaWlasnosci_Symbol"].apply(
            kategoria_wlasnosci
        )
        return df


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stabilny transformer zmiennych liczbowych:
    - tworzy cechy ciągłe (log, sqrt, kwadrat, relacje, wskaźniki),
    - tworzy cechy kategoryczne (kwantyle, przedziały),
    - brak braków danych po transformacji,
    - zachowuje nazwy kolumn dla get_feature_names_out.
    """

    def fit(self, X, y=None):
        self.fitted_ = True
        X = X.copy()
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.original_features_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        new_features = {}

        # --- Bezpieczne logi i potęgi ---
        for col in self.numeric_features_:
            vals = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            safe_vals = vals.copy()
            safe_vals[safe_vals < 0] = 0  # log1p wymaga >=0
            new_features[f"log_{col}"] = np.log1p(safe_vals)
            new_features[f"{col}_squared"] = vals**2
            new_features[f"{col}_sqrt"] = np.sqrt(np.abs(vals))

        # --- Relacje względem aktywów ---
        if "Aktywa" in X.columns:
            assets = X["Aktywa"].replace(0, np.nan).fillna(1)
            for col in self.numeric_features_:
                if col != "Aktywa":
                    new_features[f"{col}_to_assets"] = X[col] / assets

        # --- Bezpieczne dzielenie ---
        def safe_div(a, b):
            b = b.replace(0, np.nan).fillna(1)
            return a / b

        # --- Wskaźniki finansowe ---
        if {"Zysk_netto", "przychody_sprzedazy"}.issubset(X.columns):
            pm = safe_div(X["Zysk_netto"], X["przychody_sprzedazy"])
            new_features["profit_margin"] = pm

        if {"Zysk_operacyjny", "Aktywa"}.issubset(X.columns):
            roa = safe_div(X["Zysk_operacyjny"], X["Aktywa"])
            new_features["ROA"] = roa

        if {"Zysk_netto", "Kapital_wlasny"}.issubset(X.columns):
            roe = safe_div(X["Zysk_netto"], X["Kapital_wlasny"])
            new_features["ROE"] = roe

        # --- Płynność i zadłużenie ---
        if {"Aktywa_obrotowe", "Zobowiazania_krotkoterminowe"}.issubset(X.columns):
            cr = safe_div(X["Aktywa_obrotowe"], X["Zobowiazania_krotkoterminowe"])
            new_features["current_ratio"] = cr

        if {"Zobowiazania", "Kapital_wlasny"}.issubset(X.columns):
            de = safe_div(X["Zobowiazania"], X["Kapital_wlasny"])
            new_features["debt_to_equity"] = de

        # --- Cash flow ---
        if {"RP_przeplywy_operacyjne", "Zobowiazania_krotkoterminowe"}.issubset(
            X.columns
        ):
            new_features["cf_to_liabilities"] = safe_div(
                X["RP_przeplywy_operacyjne"], X["Zobowiazania_krotkoterminowe"]
            )

        if {"RP_przeplywy_operacyjne", "Aktywa"}.issubset(X.columns):
            new_features["cf_to_assets"] = safe_div(
                X["RP_przeplywy_operacyjne"], X["Aktywa"]
            )

        # --- WSKAŹNIKI ROTACJI (Obrotowości) ---
        if {"koszty_sprzedanych_produktow", "Zapasy"}.issubset(X.columns):
            # Rotacja Zapasów: Koszt Sprzedanych Produktów / Zapasy
            new_features["inventory_turnover"] = safe_div(
                X["koszty_sprzedanych_produktow"], X["Zapasy"]
            )

        if {"przychody_sprzedazy", "Naleznosci_krotkoterminowe"}.issubset(X.columns):
            # Rotacja Należności: Przychody ze Sprzedaży / Należności Krótkoterminowe
            new_features["receivables_turnover"] = safe_div(
                X["przychody_sprzedazy"], X["Naleznosci_krotkoterminowe"]
            )

        # --- KOMPONENTY ALTMAN Z-SCORE (X1, X3, X4, X5) ---

        # X1: Kapitał obrotowy netto / Aktywa (Working Capital / Total Assets)
        if {"Aktywa_obrotowe", "Zobowiazania_krotkoterminowe", "Aktywa"}.issubset(
            X.columns
        ):
            working_capital = X["Aktywa_obrotowe"] - X["Zobowiazania_krotkoterminowe"]
            new_features["working_capital_to_assets"] = safe_div(
                working_capital, X["Aktywa"]
            )

        # X3: Zysk operacyjny (EBIT) / Aktywa (EBIT / Total Assets)
        if {"Zysk_operacyjny", "Aktywa"}.issubset(X.columns):
            new_features["EBIT_to_assets"] = safe_div(X["Zysk_operacyjny"], X["Aktywa"])

        # X4: Kapitał własny / Zobowiązania (Book Value of Equity / Total Liabilities)
        if {"Kapital_wlasny", "Zobowiazania_rezerwy"}.issubset(X.columns):
            new_features["equity_to_liabilities"] = safe_div(
                X["Kapital_wlasny"], X["Zobowiazania_rezerwy"]
            )

        # X5: Przychody ze sprzedaży / Aktywa (Sales / Total Assets)
        if {"przychody_sprzedazy", "Aktywa"}.issubset(X.columns):
            new_features["sales_to_assets"] = safe_div(
                X["przychody_sprzedazy"], X["Aktywa"]
            )

        # --- Normalizacje dzienne ---
        if "wsk_liczba_dni_istnienia" in X.columns:
            days = X["wsk_liczba_dni_istnienia"].replace(0, 1)
            for col in ["Aktywa", "Zysk_netto", "Kapital_wlasny"]:
                if col in X.columns:
                    new_features[f"{col}_per_day"] = X[col] / days

        # --- Interakcje ---
        inter_pairs = [
            ("Zysk_operacyjny", "Aktywa"),
            ("Zysk_netto", "Kapital_wlasny"),
            ("Aktywa", "Kapital_wlasny"),
        ]
        for a, b in inter_pairs:
            if {a, b}.issubset(X.columns):
                new_features[f"{a}_x_{b}"] = X[a] * X[b]
                new_features[f"{a}_div_{b}"] = safe_div(X[a], X[b])

        # --- Łączenie wyników i uzupełnianie braków ---
        new_df = pd.DataFrame(new_features, index=X.index)
        new_df = new_df.replace([np.inf, -np.inf], 0).fillna(0)
        X_out = pd.concat([X, new_df], axis=1).clip(lower=-1e15, upper=1e15)

        # zapamiętaj kolumny po transformacji
        self.generated_features_ = list(new_df.columns)
        self.output_features_ = list(X_out.columns)

        return X_out

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "output_features_"):
            return np.array(self.output_features_)
        elif input_features is not None:
            return np.array(input_features)
        else:
            raise AttributeError(
                "FinancialFeatureEngineer: musisz wywołać fit/transform przed get_feature_names_out"
            )


def main():
    pass


if __name__ == "__main__":
    main()
