import functools
import io
import logging
import pathlib
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import (
    # accuracy_score,
    classification_report,
    # confusion_matrix,
    # f1_score,
    mean_absolute_error,
    mean_squared_error,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    # precision_score,
    # recall_score,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier

# typ pro Streamlit kontejnery
StContainer = st.delta_generator.DeltaGenerator


# adresar s daty
DATA_DIR = pathlib.Path("data")

# slovnik s daty (verze upravy zdrojovych dat)
# postupne bude rozsiren o zdrojova data, data s vyjmutymi sloupci,
# data vyporadana s chybejicimi hodnotami, preskalovana data, data s dummies
# → podle toho, co si uzivatel v dashboardu naklika
DATA = {}

# ----- REGRESE -----
# slovnik s nazvy modelu pro regresi
# u kazdeho modelu je treba definovat class - tridu, ktera se pouzije
# a hyperparams, ktery obsahuje slovnik nazvu hyperparametru a funkci pro
# vytvoreni streamlit widgetu, predpoklada se, ze tridy maji scikit-learn API
REGRESSION_MODELS = {
    "Linear Regression": {
        "class": LinearRegression,
        "hyperparams": {},
    },
    "Lasso": {
        "class": Lasso,
        "hyperparams": {
            "alpha": functools.partial(st.slider, "alpha", 0.0, 1.0, 0.0)},
    },
    "SVR": {
        "class": SVR,
        "hyperparams": {
            "kernel": functools.partial(
                st.selectbox,
                "kernel",
                ["linear", "poly", "rbf", "sigmoid"],
                index=2
            ),
            "C": functools.partial(st.number_input, "C", 0.0, None, 1.0),
        },
    },
}

# nazvy regresnich metrik a prislusne funkce pro vypocet
REGRESSION_METRICS = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "R2": r2_score
}

# ----- KLASIFIKACE -----
# slovnik s nazvy modelu pro klasifikaci
# u kazdeho modelu je treba definovat class - tridu, ktera se pouzije
# a hyperparams, ktery obsahuje slovnik nazvu hyperparametru a funkci pro
# vytvoreni streamlit widgetu, predpoklada se, ze tridy maji scikit-learn API
CLASSIFICATION_MODELS = {
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "hyperparams": {
            "criterion": functools.partial(
                st.selectbox,
                "criterion",
                ['gini', 'entropy']
            ),
            "max_depth": functools.partial(
                st.select_slider,
                "max depth",
                options=[2, 4, 6, 8, 10, 12]
            )
        }
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "hyperparams": {
            "C": functools.partial(
                st.number_input,
                "C (Regularization parameter)",
                0.01,
                10.0,
                step=0.01
            ),
            "max_iter": functools.partial(
                st.slider,
                "maximum number of iterations",
                100,
                500,
            )
        }
    },
    "Nearest neighbors": {
        "class": KNeighborsClassifier,
        "hyperparams": {
            "leaf_size": functools.partial(
                st.select_slider,
                "leaf size",
                options=list(range(1, 50))
            ),
            "n_neighbors": functools.partial(
                st.select_slider,
                "number of neighbors",
                options=list(range(1, 30))
            ),
            "p": functools.partial(
                st.select_slider,
                "p",
                options=[1, 2]
            )
        }
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "hyperparams": {
            "n_estimators": functools.partial(
                st.number_input,
                "the number of trees in the forest",
                100,
                5000,
                step=10
            ),
            "max_depth": functools.partial(
                st.select_slider,
                "the maximum depth of the tree",
                options=list(range(1, 20))
            ),
            "bootstrap": functools.partial(
                st.radio,
                "bootstrap samples when building trees",
                ('True', 'False')
            )
        }
    },
    "SVC": {
        "class": SVC,
        "hyperparams": {
            "C": functools.partial(
                st.number_input,
                "C (Regularization parameter)",
                0.01,
                10.0,
                step=0.01,
            ),
            "kernel": functools.partial(
                st.radio,
                "kernel",
                ("rbf", "linear")
            ),
            "gamma": functools.partial(
                st.radio,
                "gamma (Kernel Coefficient)",
                ("scale", "auto")
            )
        }
    }
}

# nazvy klasifikacnich metrik a prislusne funkce pro vypocet
CLASSIFICATION_METRICS_TO_PLOT = {
    "Matice záměn": plot_confusion_matrix,
    "Precision-Recall křivka": plot_precision_recall_curve,
    "ROC křivka": plot_roc_curve,
}


@st.cache
def load_data(
    csv_file: Union[str, pathlib.Path, io.IOBase]
) -> pd.DataFrame:
    """Funkce nacte data"""
    return pd.read_csv(csv_file, index_col=0)


@st.cache
def preprocess(
    data: pd.DataFrame,
    drop_columns: Optional[List] = None,
) -> pd.DataFrame:
    """Funkce odstrani sloupce z dateframu"""
    if drop_columns:
        data2 = data.drop(columns=drop_columns)
        return data2


@st.cache
def replace_to_na(
    data: pd.DataFrame,
    list_of_characters: List,
) -> pd.DataFrame:
    """Funkce nahradi hodnoty z list_of_characters chybejicimi hodnotami"""
    data2 = data
    for character in list_of_characters:
        data2.replace({character: np.nan}, inplace=True)
    return data2


@st.cache
def replace_na(
    data: pd.DataFrame,
    method_for_replacing_na: str,
    replace_other_na_ffill: bool,
    value_for_replacing_na: Optional[List] = None,
) -> pd.DataFrame:
    """
    Funkce nahradi chybejici hodnoty jednou z moznosti: interpolaci,
    zopakovanim hodnoty z předchoziho anebo nadchazejiciho radku, nejcasteji
    se vyskytujici hodnotou, medianem, prumerem ci vlastni hodnotou.
    Rovnez umoznuje po zvoleni jednou z vyse uvedenych moznosti doplnit
    ostatni (nehlede na datovy typ sloupce: s numerickymi ci nenumerickymi
    hodnotami) zopakovanim hodnoty z predchoziho radku
    """
    data2 = data
    if method_for_replacing_na == "interpolace":
        list_of_columns_with_float_and_int_dtypes = data2.select_dtypes(
            include=["float", 'int']).columns.tolist()
        for column in data2:
            if column in list_of_columns_with_float_and_int_dtypes:
                data2[column] = data2[column].interpolate()
    elif method_for_replacing_na == "zopakování hodnoty z předchozího řádku":
        data2 = data2.fillna(method="ffill")
    elif method_for_replacing_na == "zopakování hodnoty z nadcházejícího řádku":  # noqa
        data2 = data2.fillna(method="bfill")
    elif method_for_replacing_na == "vložení vlastní hodnoty":
        data2 = data2.fillna(value_for_replacing_na)
    elif method_for_replacing_na == "nejčastěji se vyskytující hodnota":
        data2 = data2.fillna(data2.mode())
    elif method_for_replacing_na == "medián":
        data2 = data2.fillna(data2.median())
    elif method_for_replacing_na == "průměr":
        data2 = data2.fillna(data2.mean())

    if replace_other_na_ffill:
        return data2.interpolate(method='pad')
    else:
        return data2


@st.cache
def drop_rows(
    data: pd.DataFrame,
    columns_for_dropna_rows: Optional[List] = None,
)-> pd.DataFrame:
    """Funkce vyjme radky s chybejicimi hodnotami"""
    data2 = data
    for col in columns_for_dropna_rows:
        data2 = data2.dropna(axis="rows", subset=[col])
    return data2


def scaling(
    data: pd.DataFrame,
    columns_to_scaling: Optional[List] = None,
) -> pd.DataFrame:
    """Funkce preskaluje sloupce ze seznamu columns_to_scaling"""
    if columns_to_scaling:
        data2 = data.copy()
        transformace = StandardScaler()
        # zkusime preskalovat sloupec/sloupce
        try:
            data2[columns_to_scaling] = transformace.fit_transform(
                data2[columns_to_scaling])
            return data2
        except Exception as value_error:
            # v pripade chyby ukazeme uzivateli co se stalo
            st.error(f"Chyba při přeškálování dat: {value_error}")
            # a nebudeme uz nic dalsiho zobrazovat
            return pd.DataFrame()


@st.cache
def dummies(
    data: pd.DataFrame,
    columns_to_convert: Optional[List] = None,
    get_dummies: bool = False
) -> pd.DataFrame:
    """
    Funkce prevede kategorialni hodnoty na sloupce s 0 a 1
    One-hot encoding provede i na sloupcich se dvema hodnotami
    (at jiz jsou numericke, bool ci jine)
    """
    data2 = data.copy()
    if columns_to_convert:
        for item in columns_to_convert:
            # .sort_index() proto, kdyby byly ve sloupci bool hodnoty,
            # kdyz je seradim, tak False bude 1. a nahrazen tudiz nulou
            data2 = data2.replace({
                data2[item].value_counts().sort_index().index[0]: 0,
                data2[item].value_counts().sort_index().index[1]: 1
                })
    if get_dummies:
        data2 = pd.get_dummies(data2, dtype="int64")
    return data2


def find_columns_with_numeric_values(
    col1: StContainer,
    learning_data: pd.DataFrame,
    key: str,
) -> bool:
    """
    Vraci True/False na zaklade toho,
    zda sloupce obsahuji pouze numericke hodnoty ci nikoliv
    """
    with col1.beta_container():
        if len(learning_data.select_dtypes(include=[
                "float", "int"]).columns) != len(learning_data.columns):
            st.info(
                "Dataframe obsahuje sloupce i "
                "s jinými než numerickými hodnotami"
            )
            only_columns_with_numeric_values = st.checkbox(
                "Pracovat pouze se sloupci s numerickými hodnotami",
                value=True,
                key=key
            )
        else:
            only_columns_with_numeric_values = False

        return only_columns_with_numeric_values


def find_raws_with_nan(
    col1: StContainer,
    learning_data: pd.DataFrame,
    key: str,
) -> bool:
    """
    Vraci True/False na zaklade toho,
    zda radky obsahuji chybejici hodnoty ci nikoliv
    """
    with col1.beta_container():
        if learning_data.isnull().sum().sum() > 0:
            st.info(
                "Dataframe obsahuje řádky s chybějícími hodnotami"
            )
            only_raws_without_nan = st.checkbox(
                "Pracovat pouze s řádky bez chybějících hodnot",
                value=True,
                key=key
            )
        else:
            only_raws_without_nan = False

        return only_raws_without_nan


def pca(
    col1: StContainer,
    col2: StContainer,
    learning_data: pd.DataFrame,
    source_data: pd.DataFrame,
    only_columns_with_numeric_values: bool,
    only_raws_without_nan: bool,
) -> None:
    """Analyza hlavnich komponent (PCA) v dashboardu"""

    data_for_pca = learning_data

    # pracujeme pouze se sloupci s numerickymi hodnotami
    if only_columns_with_numeric_values:
        data_for_pca = data_for_pca.select_dtypes(include=["float", "int"])

    # pracujeme pouze s radky bez chybejicich hodnot
    if only_raws_without_nan:
        data_for_pca = data_for_pca.dropna()

    # zkusime vypocitat PCA souradnice
    pca = PCA(n_components=2)
    try:
        data_pca = pd.DataFrame(
            pca.fit_transform(data_for_pca),
            columns=["PCA1", "PCA2"]
        )
    except Exception as value_error:
        # v pripade chyby ukazeme uzivateli co se stalo
        st.error(f"Chyba při výpočtu PCA souřadnic: {value_error}")
        # a nebudeme uz nic dalsiho zobrazovat
        return

    loadings = pd.DataFrame(
        pca.components_.T,
        index=data_for_pca.columns,
        columns=data_pca.columns
    )

    with col1.beta_expander(
            "Zobrazeni dataframů"):
        st.write("Výstup výpočtu PCA souřadnic")
        st.write(data_pca)
        st.write("Interpretace koeficientů")
        st.write(loadings)

    with col1.beta_expander("Nastavení grafu"):
        # vyber sloupce pro barevne rozliseni bodu v grafu dle kategorie
        interesting_column_pca_color = st.selectbox(
            "Přiřazení barvy dle kategorie", source_data.columns)
        # vyber sloupce pro velikostni rozliseni bodu v grafu dle kategorie
        interesting_column_pca_size = st.selectbox(
            "Přiřazení velikosti dle kategorie "
            "(jen sloupce s numerickými hodnotami)",
            source_data.select_dtypes(include=["float", "int"]).columns
        )
        # vyber sloupce pro zobrazeni popisku, ktery se zobrazi po najeti
        # mysi na bod v grafu
        interesting_column_pca_hover_name = st.selectbox(
            "Přiřazení titulku dle kategorie, "
            "který se zobrazí při najetí myší",
            source_data.columns
        )

    col2.header(f"Výsledek Analýzy hlavních komponent (PCA)")
    col2.write(
        "Podíl zachycené variability: "
        f"osa x: {pca.explained_variance_ratio_[0] * 100:.2f}%, "
        f"osa y: {pca.explained_variance_ratio_[1] * 100:.2f}% "
    )
    # zkusime zobrazit PCA graf
    try:
        fig = px.scatter(
            x=data_pca.PCA1,
            y=data_pca.PCA2,
            color=source_data[interesting_column_pca_color],
            size=source_data[interesting_column_pca_size],
            hover_name=source_data[interesting_column_pca_hover_name],
        )
        col2.write(fig)
    except Exception as value_error:
        # v pripade chyby ukazeme uzivateli co se stalo
        st.error(f"Chyba při zobrazování PCA grafu: {value_error}")
        # a nebudeme uz nic dalsiho zobrazovat
        return


def args_for_train_test_split(
    col1: StContainer,
    source_data: pd.DataFrame,
    key: str,
) -> Tuple:
    """
    Funkce vraci n-tici obsahujici argumenty (pomer testovaci sady
    a stratify) ktery si uzivatel naklika v ramci streamlit widgetu
    """
    # ----- TRENOVACI A TESTOVACI DATA -----
    with col1.beta_container():
        st.subheader("Trénovací a testovací data")

    with col1.beta_expander(
            "Nastavení rozdělení na testovací a trénovací data"):
        # velikost testovaci sady
        test_size = st.slider(
            "Poměr testovací sady",
            0.0, 1.0, 0.25, 0.05,
            key=key
        )

        # nastaveni stratify
        stratify_column = st.selectbox(
            "Stratify",
            [None] + list(source_data.columns),
            key=key
        )

        if stratify_column is not None:
            stratify = source_data[stratify_column]
        else:
            stratify = None

    return test_size, stratify


def regression(
    col1: StContainer,
    col2: StContainer,
    learning_data: pd.DataFrame,
    target: str,
    test_size: float,
    stratify: str,
    only_columns_with_numeric_values: bool,
    only_raws_without_nan: bool,
) -> None:
    """Regrese v dashboardu"""

    # pracujeme pouze se sloupci s numerickými hodnotami
    if only_columns_with_numeric_values:
        learning_data = learning_data.select_dtypes(include=["float", "int"])

    # pracujeme pouze s radky bez chybejicich hodnot
    if only_raws_without_nan:
        learning_data = learning_data.dropna()

    # rozdeleni na trenovaci a testovaci data
    y = learning_data[target]
    X = learning_data.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify)

    # ----- REGRESE: VYBER MODELU -----
    with col1.beta_container():
        st.subheader("Výběr modelu")

    with col1.beta_expander("Nastavení výběru modelu"):
        model = st.selectbox("Regresní model", list(REGRESSION_MODELS))
        # hodnoty hyperparametru si ulozime do slovniku typu
        # {jmeno hyperparametru: hodnota}
        hyperparams = {
            hyperparam: widget() for hyperparam, widget in REGRESSION_MODELS[
                model]["hyperparams"].items()
        }

    # ----- REGRESE: VYBER METRIKY -----
    with col1.beta_container():
        st.subheader("Výběr metriky")

    with col1.beta_expander("Nastavení výběru metriky"):
        metric = st.selectbox("Regresní metrika", list(REGRESSION_METRICS))

    # REGRESSION_MODELS[model]["class"] vraci tridu regresoru, napr.
    # LinearRegression ve slovniku hyperparams mame ulozene hodnoty
    # hyperparametru od uzivatele takto tedy muzeme vytvorit prislusny regresor
    regressor = REGRESSION_MODELS[model]["class"](**hyperparams)
    # zkusime natrenovat model
    try:
        regressor.fit(X_train, y_train)
    except Exception as prediction_error:
        # v pripade chyby ukazeme uzivateli, co se stalo
        st.error(f"Chyba při fitování regresního modelu: {prediction_error}")
        # a nebudeme uz nic dalsiho zobrazovat
        return

    # predikce pomoci natrenovaneho modelu
    y_predicted = regressor.predict(X_test)
    prediction_error = REGRESSION_METRICS[metric](y_predicted, y_test)

    col2.header(f"Výsledek regresního modelu {model}")
    col2.write(f"{metric}: {prediction_error:.3g}")

    # vytvorime pomocny dataframe se sloupcem s predikci
    predicted_target_column = f"{target} - predicted"
    complete_data = learning_data.assign(**{
        predicted_target_column: regressor.predict(X)})

    # vykreslime spravne vs predikovane body
    fig = px.scatter(complete_data, x=target, y=predicted_target_column)
    # pridame caru ukazujici idealni predikci
    fig.add_trace(
        go.Scatter(
            x=[complete_data[target].min(), complete_data[target].max()],
            y=[complete_data[target].min(), complete_data[target].max()],
            mode="lines",
            line=dict(width=2, color="DarkSlateGrey"),
            name="ideal prediction",
        )
    )
    col2.write(fig)


def classification(
    col1: StContainer,
    col2: StContainer,
    learning_data: pd.DataFrame,
    target_column: str,
    target_value: str,
    test_size: float,
    stratify: str,
    only_columns_with_numeric_values: bool,
    only_raws_without_nan: bool,
) -> None:
    """Binarni klasifikace v dashboardu"""

    # pracujeme pouze s radky bez chybejicich hodnot
    if only_raws_without_nan:
        learning_data = learning_data.dropna()

    # ----- BINARNI KLASIFIKACE: PRIPRAVA DAT -----
    # sloupce rozdelime na vstupni (X) a vystup s vyslednou tridou (y)

    # odezva
    # je-li vybrana hodnota ve sloupci s kategorickymi hodnotami
    if target_value:
        y = learning_data[target_column] == target_value
    # je-li vybran sloupec se dvema hodnotami (treba: 0 a 1)
    else:
        y = learning_data[target_column]

    y = y.astype(int)
    X = learning_data.drop(columns=[target_column])

    # pracujeme pouze se sloupci s numerickými hodnotami
    if only_columns_with_numeric_values:
        X = X.select_dtypes(include=["float", "int"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=0)

    # ----- BINARNI KLASIFIKACE: VYTVORENI A NAUCENI MODELU -----
    with col1.beta_container():
        st.subheader("Výběr modelu")

    with col1.beta_expander("Nastavení výběru modelu"):
        model = st.selectbox("Klasifikační model", list(CLASSIFICATION_MODELS))
        # hodnoty hyperparametru si ulozime do slovniku typu
        # {jmeno hyperparametru: hodnota}
        hyperparams = {
            hyperparam: widget() for hyperparam, widget in CLASSIFICATION_MODELS[  # noqa
                model]["hyperparams"].items()
        }

    # CLASSIFICATION_MODELS[model]["class"] vraci tridu regresoru, napr.
    # Logistic Regression ve slovniku hyperparams mame ulozene hodnoty
    # hyperparametru od uzivatele takto tedy muzeme vytvorit prislusny regresor
    classifier = CLASSIFICATION_MODELS[model]["class"](**hyperparams)

    # zkusime natrenovat model
    try:
        classifier.fit(X_train, y_train)
    except Exception as prediction_error:
        # v pripade chyby ukazeme uzivateli, co se stalo
        st.error(
            f"Chyba při fitování klasifikačního modelu: {prediction_error}")
        # a nebudeme uz nic dalsiho zobrazovat
        return

    # ----- BINARNI KLASIFIKACE: VYBER METRIKY -----
    with col1.beta_container():
        st.subheader("Vyhodnocení modelu")

    with col1.beta_expander("Nastavení vyhodnoceni modelu"):
        view_classification_report = st.checkbox(
            "Zobrazení klasifikačního reportu", value=True)
        metric_to_plot = st.selectbox(
            "Vykreslení metriky",
            list(CLASSIFICATION_METRICS_TO_PLOT)
        )

    # predikce pomoci natrenovaneho modelu
    pred = classifier.predict(X_test)

    col2.header(f"Výsledek klasifikačního modelu {model}")
    if view_classification_report:
        col2.text('Model Report:\n ' + classification_report(y_test, pred))

    col2.write(f"{metric_to_plot}:")
    if metric_to_plot == "Matice záměn":
        CLASSIFICATION_METRICS_TO_PLOT[metric_to_plot](
            classifier, X_test, y_test, cmap=plt.cm.Blues)
    else:
        CLASSIFICATION_METRICS_TO_PLOT[metric_to_plot](
            classifier, X_test, y_test)
    col2.pyplot()


def main() -> None:
    # zakladni vlastnosti aplikace: jmeno, siroke rozlozeni
    st.set_page_config(page_title="Fishboard či jiný dashboard", layout="wide")
    st.title("Fishboard anebo dashboard z vašich vlastních dat")
    # at se nezobrazuje varvani pri zobrazovani grafu pomoci st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # pouzijeme dva sloupce
    col1, col2 = st.beta_columns(2)

    # ----- VYBER DAT -----
    with col1.beta_container():
        st.header("Výběr dat")
        st.write(
            "Vstupní data jsou ze souboru fish_data.csv, "
            "ale můžete je kdykoliv změnit načtením svých vlastních dat:"
            )

    with col1.beta_expander("Načtení dat"):
        data_file_path = st.file_uploader("Data file")
        data = None
        if data_file_path is not None:
            # read data if user uploads a file
            data = pd.read_csv(data_file_path)
            # seek back to position 0 after reading
            data_file_path.seek(0)
        if data is None:
            st.warning("No data loaded")
            data = load_data(DATA_DIR / "fish_data.csv")
        source_data = data
    DATA["zdrojová data"] = source_data

    # ----- PRIPRAVA DAT -----
    with col1.beta_container():
        st.header("Příprava dat")
        st.subheader("Preprocessing")

    with col1.beta_expander("Vyjmutí sloupců"):
        drop_columns = st.multiselect("Slouce k vyjmutí", source_data.columns)
    if drop_columns:
        dropped_columns_data = preprocess(source_data, drop_columns)
        DATA["data s vyjmutými sloupci"] = dropped_columns_data
    else:
        dropped_columns_data = source_data

    # ----- CHYBEJICI HODNOTY ----- # with_missing_values_data
    with col1.beta_container():
        st.subheader("Chybějící hodnoty")

    # predavame si dataframe do nove promenne
    with_missing_values_data = dropped_columns_data

    # existuji-li chybejici hodnoty v dataframu
    if with_missing_values_data.isnull().sum().sum() > 0:
        text = "V dataframu jsou chybějící hodnoty."

        # seznam sloupcu s chybejicimi hodnotami, ktery je aktualizovan
        columns_with_na = list(with_missing_values_data.columns[
            with_missing_values_data.isnull().any()])

        # ----- NAHRAZENI ZNAKU NA CHYBEJICI HODNOTY -----
        with col1.beta_expander(
                "Nastavení nahrazení znaků na chybějící hodnoty"):
            character = st.text_input(
                "Znaky k převodu na chybějící hodnotu oddělené čárkou")
            if character:
                list_of_characters = character.split(",")
                list_of_characters = [
                    item.strip() for item in list_of_characters]
                with_missing_values_data = replace_to_na(
                    with_missing_values_data, list_of_characters)
                DATA["data vypořádaná s chybějícími hodnotami"] = with_missing_values_data  # noqa

        # neprovede-li se beta_expander (nahrazeni znaku na chybejici hodnoty),
        # predavame si puvodni dataframe do nove promenne
        replaced_na_data = with_missing_values_data

        # ----- NAHRAZENI CHYBEJICICH HODNOT -----
        with col1.beta_expander(
                "Nastavení nahrazení chybějících hodnot"):
            list_of_methods_for_replacing_na = [
                "žádná",
                "průměr",
                "medián",
                "nejčastěji se vyskytující hodnota",
                "interpolace",
                "zopakování hodnoty z předchozího řádku",
                "zopakování hodnoty z nadcházejícího řádku",
                "vložení vlastní hodnoty"
            ]
            method_for_replacing_na = st.selectbox(
                "Metoda nahrazení chybějících hodnot",
                list_of_methods_for_replacing_na
            )

            if method_for_replacing_na == "vložení vlastní hodnoty":
                value_for_replacing_na = st.number_input(
                    "Vlastní hodnota k nahrazení chybějících hodnot")
            else:
                value_for_replacing_na = None

            replace_other_na_ffill = st.checkbox(
                "Převést ostatní chybějící hodnoty na duplicitní hodnotu "
                "z předchozího řádku"
            )
            if replace_other_na_ffill:
                text = "V dataframu již nejsou žádné chybějící hodnoty."
            replaced_na_data = replace_na(
                replaced_na_data,
                method_for_replacing_na,
                replace_other_na_ffill,
                value_for_replacing_na,
            )

            # st.write(replace_other_na_ffill)

            # st.write(replaced_na_data)
            # st.write(source_data)

            # aktualizace → je-li zaskrtnuto replace_other_na_ffill
            # a neni-li metoda pro nahrazeni nan zvolena defaultni "zadna"
            if replace_other_na_ffill or (method_for_replacing_na != "žádná"):
                DATA["data vypořádaná s chybějícími hodnotami"] = replaced_na_data  # noqa
            columns_with_na = list(replaced_na_data.columns[
                replaced_na_data.isnull().any()])

        # ----- VYJMUTI RADKU S CHYBEJICIMI HODNOTAMI ----- # dropna_rows_data

        # tvorba defaultniho dataframu pro tuto sekci
        dropna_rows_data = replaced_na_data

        # existuji-li jeste chybejici hodnoty v dataframu
        # po (ne)nahrazeni chybejicich hodnot
        if replaced_na_data.isnull().sum().sum() > 0:
            text = "V dataframu jsou chybějící hodnoty."

            with col1.beta_expander(
                    "Nastavení vyjmutí řádků s chybějícími hodnotami"):
                dropna_rows = st.checkbox(
                    "Vyjmutí řádků s chybějícími hodnotami")
                columns_for_dropna_rows = st.multiselect(
                    "Sloupce obsahující chybějící hodnoty",
                    columns_with_na,
                    default=columns_with_na
                )

                # vyjmuti radku na zaklade sloupcu,
                # v nichz jsou chybejici hodnoty
                if columns_for_dropna_rows and dropna_rows:
                    dropna_rows_data = drop_rows(
                        replaced_na_data, columns_for_dropna_rows)
                    DATA["data vypořádaná s chybějícími hodnotami"] = dropna_rows_data  # noqa

                    # nachazi se v novem dataframu jeste nejake
                    # chybejici hodnoty
                    if dropna_rows_data.isnull().sum().sum() == 0:
                        text = (
                            "V dataframu již nejsou žádné "
                            "chybějící hodnoty."
                        )
                        columns_with_na = list(dropna_rows_data.columns[
                            dropna_rows_data.isnull().any()])
                    else:
                        columns_with_na = list(dropna_rows_data.columns[
                            dropna_rows_data.isnull().any()])
                else:
                    dropna_rows_data = replaced_na_data

        # neprovede-li se beta_expander (vyjmuti radku s chybejicimi
        # hodnotami), predavame si predchozi dataframe do nove promenne
        scaling_data = dropna_rows_data

        with col1.beta_expander("Přehled výskytu chybějících hodnot"):
            st.info(text)
            if columns_with_na:
                st.write(dropna_rows_data[columns_with_na].isnull().sum())

    # v puvodnim dataframu se chybejici hodnoty nenachazi
    else:
        scaling_data = dropped_columns_data

        with col1.beta_expander("Přehled výskytu chybějících hodnot"):
            st.info("V dataframu nejsou žádné chybějící hodnoty.")

    # ----- PRESKALOVANI DAT ----- # scaling_data

    with col1.beta_container():
        st.subheader("Přeškálování dat")

    with col1.beta_expander("Nastavení škálování dat"):
        # vybirame jenom sloupce s numerickymi hodnotami
        columns_to_scaling = st.multiselect(
            "Sloupce ke škálování (jen sloupce s numerickými hodnotami)",
            scaling_data.select_dtypes(include=["float", "int"]).columns
        )

        if columns_to_scaling:
            scaling_data = scaling(scaling_data, columns_to_scaling)
            DATA["přeškálovaná data"] = scaling_data

    # ----- ZPRACOVANI KATEGORICKYCH HODNOT ----- # dummies_data

    # tvorba defaultniho dataframu pro tuto sekci
    dummies_data = scaling_data

    with col1.beta_container():
        st.subheader("Převedení hodnot na čísla")

    with col1.beta_expander("Nastavení způsobu převodu"):
        # ----- SLOUPCE SE DVEMA HODNOTAMI -----
        st.subheader("Sloupce se dvěma hodnotami")
        # tvorba seznamu sloupcu majici 2 hodnoty k prevodu na numericke 0 a 1
        columns_with_two_values = []
        for item in scaling_data.columns.to_list():
            if len(scaling_data[item].value_counts()) == 2:
                columns_with_two_values.append(item)
        if columns_with_two_values:
            columns_to_convert = st.multiselect(
                "Sloupce se dvěma hodnotami k převodu na 0 a 1",
                columns_with_two_values
            )
        else:
            columns_to_convert = []
            st.info(
                "V dataframu nejsou sloupce se dvěma hodnotami "
                "k převodu na 0 a 1."
                )

        # ----- ONE-HOT ENCODING -----
        st.subheader("One-hot encoding")
        get_dummies = st.checkbox("Get dummies")

    if get_dummies or columns_to_convert:
        dummies_data = dummies(
            scaling_data, columns_to_convert, get_dummies)
        DATA["data s převedenými hodnotami na čísla"] = dummies_data

    # data pripravena k regresi, PCA, binarni klasifikaci
    learning_data = dummies_data

    # ----- ZOBRAZENI DAT -----
    with col1.beta_container():
        st.header("Zobrazení dat")
        select_displayed_data = st.radio(
            "Výběr zobrazovaných dat", list(DATA.keys()))

    with col1.beta_expander("Zobrazení dataframu"):
        displayed_data = DATA[select_displayed_data]
        st.write(
            "Počet řádků:", displayed_data.shape[0],
            ", počet sloupců:", displayed_data.shape[1]
            )
        st.dataframe(displayed_data)

    # ----- SCATTER MATRIX GRAF ----
    with col1.beta_expander("Nastavení scatter matrix grafu"):
        # vstup 2: vyber parametru scatter matrix
        dimensions = st.multiselect("Scatter matrix dimensions", list(
            displayed_data.columns), default=list(displayed_data.columns))
        color = st.selectbox("Color", displayed_data.columns)
        opacity = st.slider("Opacity", 0.0, 1.0, 0.5)

    with col2.beta_container():
        st.header("Zobrazení scatter matrix grafu")
        # scatter matrix plot
        st.write(px.scatter_matrix(
            displayed_data,
            dimensions=dimensions,
            color=color,
            opacity=opacity
            ))

    # ----- ROZDELENI DAT -----
    with col1.beta_expander("Nastavení rozdělení dat"):
        # vyber sloupce pro zobrazeni rozdeleni dat
        interesting_column = st.selectbox(
            "Interesting column", displayed_data.columns)
        # vyber funkce pro zobrazeni rozdelovaci funkce
        plot_type = {
            "Box": px.box,
            "Histogram": px.histogram,
            "Violin": px.violin,
        }
        dist_plot = st.selectbox("Plot type", list(plot_type.keys()))

    with col2.beta_container():
        st.header("Zobrazení rozdělení dat")
        # zkusime zobrazit
        try:
            st.write(plot_type[dist_plot](
                displayed_data, x=interesting_column, color=color))
        except Exception as prediction_error:
            # v pripade chyby ukazeme uzivateli co se stalo
            st.error(
                f"Chyba při zobrazování rozdělení dat: {prediction_error}"
            )
            # a nebudeme uz nic dalsiho zobrazovat
            return

    # ----- ANALYZA HLAVNICH KOMPONENT (PCA) ----- # data_pca
    with col1.beta_container():
        st.header("Analýza hlavních komponent (PCA)")
        # pracujeme jen se sloupci s numerickými hodnotami
        only_columns_with_numeric_values_I = find_columns_with_numeric_values(
            col1, learning_data, "pca_numeric_values")
        # pracujeme jen s radky bez chybejicich hodnot
        only_raws_without_nan_I = find_raws_with_nan(
            col1, learning_data, "pca_nan")

    pca(
        col1,
        col2,
        learning_data,
        source_data,
        only_columns_with_numeric_values_I,
        only_raws_without_nan_I
    )

    # ----- REGRESE -----
    with col1.beta_container():
        st.header("Regrese")
        # pracujeme jen se sloupci s numerickými hodnotami
        only_columns_with_numeric_values_II = find_columns_with_numeric_values(
            col1, learning_data, "regression_numeric_values")
        # pracujeme jen s radky bez chybejicich hodnot
        only_raws_without_nan_II = find_raws_with_nan(
            col1, learning_data, "regression_nan")

    # ----- ODEZVA -----
    with col1.beta_container():
        st.subheader("Výběr odezvy")

    with col1.beta_expander("Nastavení výběru odezvy"):
        if only_columns_with_numeric_values_II:
            target = st.selectbox(
                "Sloupec s odezvou (jen sloupce s numerickými hodnotami)",
                learning_data.select_dtypes(include=["float", "int"]).columns,
                key="regression"
            )
        else:
            target = st.selectbox(
                "Sloupec s odezvou",
                learning_data.columns,
                key="regression"
            )

    # ----- TRENOVACI A TESTOVACI DATA -----
    test_size, stratify = args_for_train_test_split(
        col1, source_data, "regression")

    # ----- SAMOTNA REGRESE -----
    regression(
        col1,
        col2,
        learning_data,
        target,
        test_size,
        stratify,
        only_columns_with_numeric_values_II,
        only_raws_without_nan_II
    )

    # ----- BINARNI KLASIFIKACE -----
    with col1.beta_container():
        st.header("Binární klasifikace")

    # ----- ZJISTUJEME, ZDA DATAFRAME OBSAHUJE SLOUPCE
    # S KATEGORICKYMI HODNOTAMI (TJ. SLOUPCE SE 2 HODNOTAMI ANEBO
    # NENUMERICKÉ SLOUPCE) -----

    # tvroba seznamu sloupcu majici dve hodnoty
    columns_with_two_values = []
    for item in learning_data.columns.to_list():
        if len(learning_data[item].value_counts()) == 2:
            columns_with_two_values.append(item)
    # tvorba seznamu sloupcu nemajici numericke hodnoty
    columns_with_no_numeric_values = list(
        set(learning_data.columns) -
        set(learning_data.select_dtypes(include=["float", 'int']).columns)
    )
    # tvorba seznamu sloupcu s kategorickymi hodnotami spojenim
    # 2 predchozich seznamu (sloupce se 2 hodnotami, nenumericke sloupce)
    categorical_columns = [
        *columns_with_two_values,
        *columns_with_no_numeric_values
    ]

    # seznam pouze unikatnich sloupcu s kategorickymi hodnotami
    categorical_columns = list(set(categorical_columns))
    categorical_columns.sort()

    # rozliseni, zda dataframe obsahuje kategoricke hodnoty
    # tj. sloupce se 2 hodnotami anebo nenumericke sloupce
    if not categorical_columns:
        with col1.beta_container():
            st.info(
                "Dataframe neobsahuje sloupce s kategorickými hodnotami "
                "(tj. jen sloupce s 2 hodnotami (vč. numerických) "
                "anebo sloupce bez numerických hodnot)"
            )
            # a dale nebudeme pro sekci BINARNI KLASIFIKACE nic zobrazovat

    else:
        # pracujeme jen s radky bez chybejicich hodnot
        only_raws_without_nan_III = find_raws_with_nan(
            col1, learning_data, "classification_nan")

        # ----- VYBER ODEZVY -----
        with col1.beta_container():
            st.subheader("Výběr odezvy")

        with col1.beta_expander("Nastavení výběru odezvy"):

            target_column = st.selectbox(
                "Sloupec s odezvou: kategorické hodnoty "
                "(jen sloupce s 2 hodnotami (vč. numerických) "
                "anebo sloupce bez numerických hodnot)",
                categorical_columns,
                key="classification"
            )

            # vybere-li si uzivatel sloupec s kategorickymi hodnotami,
            # rozbali se mu nabidka unikatnich hodnot, ktere obsahuje,
            # a vybere si jednu z nich jako odezvu
            if target_column not in columns_with_two_values:
                target_value = st.selectbox(
                    "Specifikace kategorické hodnoty",
                    learning_data[target_column].unique(),
                    key="classification"
                )
            else:
                target_value = ""

        # pracujeme jen se sloupci s numerickými hodnotami
        only_columns_with_numeric_values_III = find_columns_with_numeric_values(  # noqa
            col1,
            learning_data.drop(columns=[target_column]),
            "classification_numeric_values"
        )

        # ----- TRENOVACI A TESTOVACI DATA -----
        test_size, stratify = args_for_train_test_split(
            col1, source_data, "classification")

        # ----- SAMOTNA BINARNI KLASIFIKACE -----
        classification(
            col1,
            col2,
            learning_data,
            target_column,
            target_value,
            test_size,
            stratify,
            only_columns_with_numeric_values_III,
            only_raws_without_nan_III
        )


if __name__ == "__main__":
    logging.basicConfig()
    main()
