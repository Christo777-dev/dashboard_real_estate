"""Einfaches Preismodell für Airbnb-Inserate (Random Forest).

Merkmale: Stadt, Anzahl der Schlafzimmer, Alter des Inserats in Monaten.
Das Modell lernt auf dem Logarithmus des Preises, weil Preise stark rechtsschief
verteilt sind; Vorhersagen werden zurück in Dollar umgerechnet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

TOP_N_CITIES = 40
OTHER_CITY = "Andere Stadt in Texas"
MAX_BEDROOMS = 8


def _features(
    city: pd.Series, bedrooms: pd.Series, age_months: pd.Series, cities: list[str]
) -> pd.DataFrame:
    city_cat = pd.Categorical(
        city.where(city.isin(cities), OTHER_CITY), categories=cities + [OTHER_CITY]
    )
    X = pd.get_dummies(city_cat, prefix="city", dtype=float)
    X.index = city.index
    X["bedrooms"] = bedrooms.clip(upper=MAX_BEDROOMS).astype(float)
    X["age_months"] = age_months.astype(float)
    return X


@dataclass
class PriceModel:
    model: RandomForestRegressor
    cities: list[str]
    typical_error: float  # mittlere Abweichung in Dollar (Testdaten)
    median_pct_error: float  # mittlere prozentuale Abweichung (Median)
    r2: float
    ratio_low: float  # 10. Perzentil von Ist/Prognose
    ratio_high: float  # 90. Perzentil von Ist/Prognose
    n_train: int
    n_test: int

    def predict(self, city: str, bedrooms: int, age_months: int) -> tuple[float, float, float]:
        X = _features(
            pd.Series([city]), pd.Series([bedrooms]), pd.Series([age_months]), self.cities
        )
        price = float(np.exp(self.model.predict(X)[0]))
        return price, price * self.ratio_low, price * self.ratio_high


def train_price_model(df: pd.DataFrame) -> PriceModel:
    """Trainiert das Modell auf Inseraten bis zum 99. Preisperzentil."""
    data = df[df["price"] <= df["price"].quantile(0.99)]
    cities = data["city"].value_counts().index[:TOP_N_CITIES].tolist()
    X = _features(data["city"], data["bedrooms"], data["age_months"], cities)
    y = np.log(data["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    model.set_params(n_jobs=1)  # Einzelprognosen sind ohne Thread-Start schneller
    predicted = np.exp(model.predict(X_test))
    actual = np.exp(y_test.to_numpy())
    ratio = actual / predicted
    return PriceModel(
        model=model,
        cities=cities,
        typical_error=float(np.mean(np.abs(actual - predicted))),
        median_pct_error=float(np.median(np.abs(ratio - 1))),
        r2=float(r2_score(y_test, model.predict(X_test))),
        ratio_low=float(np.quantile(ratio, 0.1)),
        ratio_high=float(np.quantile(ratio, 0.9)),
        n_train=len(X_train),
        n_test=len(X_test),
    )
