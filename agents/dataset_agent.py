import pandas as pd


class DatasetAgent:
    def detect_id_columns(self, df):

        id_cols = []

        for col in df.columns:

            unique_ratio = df[col].nunique() / len(df)

            if unique_ratio > 0.95 and "id" in col.lower():
                id_cols.append(col)

            elif unique_ratio > 0.99:
                id_cols.append(col)

        return id_cols

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

        constant_columns = []

        for col in df.columns:

            if df[col].nunique() == 1:
                constant_columns.append(col)

        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "missing_values": missing_values,
            "constant_columns": constant_columns
        }

    def detect_target(self, df):

        common_targets = [
            "target",
            "label",
            "class",
            "attrition",
            "churn",
            "saleprice"
        ]

        for col in df.columns:

            if col.lower() in common_targets:
                return col

        return df.columns[-1]

    def detect_task(self, target_series):

        if target_series.nunique() <= 10:
            return "classification"

        return "regression"