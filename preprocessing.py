"""Data loading, encoding, and model training for the German Credit study."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from feature_metadata import build_feature_metadata

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLUMNS = [
    "Status_of_existing_checking_account", "Duration_in_month", "Credit_history", "Purpose",
    "Credit_amount", "Savings_account_bonds", "Present_employment_since",
    "Installment_rate_in_percentage_of_disposable_income", "Personal_status_and_sex",
    "Other_debtors_guarantors", "Present_residence_since", "Property", "Age_in_years",
    "Other_installment_plans", "Housing", "Number_of_existing_credits_at_this_bank", "Job",
    "Number_of_people_being_liable_to_provide_maintenance_for", "Telephone", "Foreign_worker", "Target",
]


@dataclass
class GermanCreditData:
    pipeline: Pipeline
    X_train_encoded: np.ndarray
    X_test_encoded: np.ndarray
    y_test: np.ndarray
    predicted_classes: np.ndarray
    classes: list[int]
    encoded_feature_names: list[str]
    metadata: dict
    class_labels: list[str] | None = None


def load_data(url: str = DATA_URL):
    data = pd.read_csv(url, sep=" ", header=None, names=COLUMNS)
    data["Target"] = data["Target"].map({1: 1, 2: 0})
    return data.drop(columns="Target"), data["Target"]


def train_model(url: str = DATA_URL, random_state: int = 42) -> GermanCreditData:
    X, y = load_data(url)
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=random_state)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    pipeline.fit(X_train, y_train)
    pre = pipeline.named_steps["preprocessor"]
    train_encoded, test_encoded = pre.transform(X_train), pre.transform(X_test)
    if hasattr(train_encoded, "toarray"):
        train_encoded = train_encoded.toarray()
        test_encoded = test_encoded.toarray()
    names = list(pre.get_feature_names_out())
    model = pipeline.named_steps["classifier"]
    return GermanCreditData(
        pipeline, np.asarray(train_encoded), np.asarray(test_encoded), y_test.to_numpy(),
        model.predict(test_encoded), [int(c) for c in model.classes_], names,
        build_feature_metadata(pre, categorical, numerical),
    )


def train_model_from_dataframe(frame: pd.DataFrame, target: str, features: list[str], random_state: int = 42):
    """Train the same analysis model for a user-provided classification dataset."""
    selected = frame[features].copy()
    labels = frame[target].astype(str).str.strip()
    valid = labels.ne("") & labels.notna()
    selected, labels = selected.loc[valid], labels.loc[valid]
    for column in selected.columns:
        converted = pd.to_numeric(selected[column], errors="coerce")
        non_empty = selected[column].notna() & selected[column].astype(str).str.strip().ne("")
        if non_empty.any() and converted[non_empty].notna().all():
            selected[column] = converted
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    categorical = selected.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = [column for column in selected.columns if column not in categorical]
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([( "imputer", SimpleImputer(strategy="median"))]), numerical),
    ])
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=120, random_state=random_state, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(selected, y, test_size=0.3, random_state=random_state, stratify=y)
    pipeline.fit(X_train, y_train)
    pre = pipeline.named_steps["preprocessor"]
    train_encoded, test_encoded = pre.transform(X_train), pre.transform(X_test)
    if hasattr(train_encoded, "toarray"):
        train_encoded, test_encoded = train_encoded.toarray(), test_encoded.toarray()
    model = pipeline.named_steps["classifier"]
    return GermanCreditData(
        pipeline, np.asarray(train_encoded), np.asarray(test_encoded), np.asarray(y_test), model.predict(test_encoded),
        [int(c) for c in model.classes_], list(pre.get_feature_names_out()),
        build_feature_metadata(pre, categorical, numerical), [str(label) for label in encoder.classes_],
    )
