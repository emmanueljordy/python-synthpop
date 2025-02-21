import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fancyimpute import IterativeImputer  # For MICE and EM
import warnings

class MissingDataHandler:
    """Detects missingness type (MCAR, MAR, MNAR) and applies automatic imputation."""

    def __init__(self):
        self.imputers = {}

    def detect_missingness(self, df: pd.DataFrame) -> dict:
        """Detects missingness type for each column.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            dict: Dictionary mapping column names to detected missingness type.
        """
        missingness = {}

        for col in df.columns:
            missing_values = df[col].isna().sum()
            if missing_values == 0:
                continue  # No missing values → Skip detection

            # 1️⃣ Little's MCAR Test
            _, p_value = stats.chisquare(df[col].dropna().value_counts())
            if p_value > 0.05:
                missingness[col] = "MCAR"
                continue

            # 2️⃣ Logistic Regression (MAR Detection)
            missing_mask = df[col].isna().astype(int)
            observed_data = df.drop(columns=[col]).fillna(df.mean())
            
            model = LogisticRegression()
            model.fit(observed_data, missing_mask)
            if model.score(observed_data, missing_mask) > 0.6:  # Predictable missingness → MAR
                missingness[col] = "MAR"
                continue

            # 3️⃣ Distributional Check (MNAR Detection)
            observed_values = df[col].dropna()
            missing_rows = df[col].isna()
            if missing_rows.sum() > 0:
                missing_values = df.loc[missing_rows, df.columns != col].mean(axis=1)
                _, p_value = stats.ks_2samp(observed_values, missing_values)
                if p_value < 0.05:
                    missingness[col] = "MNAR"
                    continue

            missingness[col] = "MAR"  # Default to MAR if uncertain

        return missingness

    def apply_imputation(self, df: pd.DataFrame, missingness: dict) -> pd.DataFrame:
        """Automatically applies imputation based on missingness type.

        Args:
            df (pd.DataFrame): Input data with missing values.
            missingness (dict): Mapping of column names to missingness type.

        Returns:
            pd.DataFrame: Data with imputed values.
        """
        df = df.copy()

        for col, mtype in missingness.items():
            if df[col].dtype == "object":
                # Categorical Data
                if mtype == "MCAR":
                    df[col].fillna(df[col].mode()[0], inplace=True)  # Mode Imputation
                elif mtype == "MAR":
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    df[col] = IterativeImputer().fit_transform(df[[col]])  # Classification-based
                elif mtype == "MNAR":
                    df[col].fillna("Missing", inplace=True)  # Add "Missing" Category

            else:
                # Numerical Data
                if mtype == "MCAR":
                    df[col] = SimpleImputer(strategy="mean").fit_transform(df[[col]])
                elif mtype == "MAR":
                    df[col] = IterativeImputer().fit_transform(df[[col]])  # Regression-based
                elif mtype == "MNAR":
                    df[col] = IterativeImputer().fit_transform(df[[col]])  # EM Algorithm

        return df
