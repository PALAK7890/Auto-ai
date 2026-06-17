import pandas as pd


class DatasetAgent:

    def analyze(self, df):

        numerical_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_cols = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        missing_values = (
            df.isnull()
            .sum()
            .to_dict()
        )

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "missing_values": missing_values
        }

    def detect_task(self, target_series):

        if target_series.nunique() <= 10:
            return "classification"

        return "regression"