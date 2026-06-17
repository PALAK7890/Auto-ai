from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

class PreprocessingAgent:

    def process(
        self,
        df,
        target,
        analysis
    ):

        X = df.drop(columns=[target])
        y = df[target]

        # Remove constant columns
        X = X.drop(
            columns=analysis["constant_columns"],
            errors="ignore"
        )

        # Remove ID columns
        X = X.drop(
            columns=analysis["id_columns"],
            errors="ignore"
        )

        # Handle missing values
        for col in X.columns:

            if is_numeric_dtype(
                    X[col]
            ):

                X[col] = X[col].fillna(
                    X[col].median()
                )

            else:

                X[col] = X[col].fillna(
                    X[col].mode()[0]
                )

        # Encode categoricals
        for col in X.columns:

            if not is_numeric_dtype(
                X[col]
            ):

                encoder = LabelEncoder()

                X[col] = encoder.fit_transform(
                    X[col].astype(str)
                )

        # Encode target
        if not is_numeric_dtype(y):

            y = (
                LabelEncoder()
                .fit_transform(y)
            )

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )