from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from databricks.automl_runtime.sklearn import OneHotEncoder as DBOneHotEncoder

import pandas as pd

# Boolean features
def get_bool_pipeline():
    bool_imputers = []  # Add real imputers here if needed
    return Pipeline(steps=[
        ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
        ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
        ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

# Numeric features
def get_numerical_pipeline():
    num_imputers = [
        ("impute_mean", SimpleImputer(), ["avg_price_increase", "monthly_charges", "num_optional_services", "tenure", "total_charges"])
    ]
    return Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ])

# Categorical features
def get_onehot_pipeline():
    one_hot_imputers = []  # Add real imputers here if needed
    return Pipeline(steps=[
        ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
        ("one_hot_encoder", DBOneHotEncoder(handle_unknown="indicator")),
    ])

# Full column transformer
def get_preprocessor():
    bool_features = ["gender", "phone_service", "dependents", "senior_citizen", "paperless_billing", "partner"]
    num_features = ["monthly_charges", "total_charges", "avg_price_increase", "tenure", "num_optional_services"]
    cat_features = ["contract", "device_protection", "internet_service", "multiple_lines", "online_backup", "online_security", "payment_method", "streaming_movies", "streaming_tv", "tech_support"]

    transformers = [
        ("boolean", get_bool_pipeline(), bool_features),
        ("numerical", get_numerical_pipeline(), num_features),
        ("onehot", get_onehot_pipeline(), cat_features),
    ]

    return ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)
