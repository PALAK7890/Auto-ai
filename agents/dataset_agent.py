import pandas as pd


class DatasetAgent:

    def analyze(self, df):

        numerical_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        missing_values = (
            df.isnull()
            .sum()
            .to_dict()
        )

        constant_columns = []

        for col in df.columns:

            if df[col].nunique() <= 1:
                constant_columns.append(col)

        id_columns = self.detect_id_columns(df)

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "missing_values": missing_values,
            "constant_columns": constant_columns,
            "id_columns": id_columns
        }

    def detect_id_columns(self, df):

        id_cols = []

        id_keywords = [
            "id",
            "number",
            "userid",
            "customerid",
            "employeeid",
            "transactionid",
            "orderid"
        ]

        for col in df.columns:

            col_lower = col.lower()

            if any(
                keyword in col_lower
                for keyword in id_keywords
            ):
                id_cols.append(col)

        return id_cols

    def detect_task(self, target_series):

        if not pd.api.types.is_numeric_dtype(
            target_series
        ):
            return "Classification"

        unique_values = target_series.nunique()

        if unique_values <= 20:
            return "Classification"

        return "Regression"
    def dataset_summary(self, df):

        return {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb":
                    round(
                        df.memory_usage(
                            deep=True
                        ).sum() / 1024**2,
                        2
                    )
            }