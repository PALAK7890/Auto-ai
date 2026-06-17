from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype


class PreprocessingAgent:

    def process(
        self,
        df,
        target,
        analysis,
        task
    ):

        # ------------------------
        # Separate Features/Target
        # ------------------------

        X = df.drop(columns=[target])
        y = df[target]

        # ------------------------
        # Remove Useless Columns
        # ------------------------

        X = X.drop(
            columns=analysis["constant_columns"],
            errors="ignore"
        )

        X = X.drop(
            columns=analysis["id_columns"],
            errors="ignore"
        )

        # ------------------------
        # Missing Values
        # ------------------------

        for col in X.columns:

            if is_numeric_dtype(X[col]):

                X[col] = X[col].fillna(
                    X[col].median()
                )

            else:

                X[col] = X[col].fillna(
                    X[col].mode()[0]
                )

        # ------------------------
        # Encode Categoricals
        # ------------------------

        for col in X.columns:

            if not is_numeric_dtype(X[col]):

                encoder = LabelEncoder()

                X[col] = encoder.fit_transform(
                    X[col].astype(str)
                )

        # ------------------------
        # Target Processing
        # ------------------------

        y_scaler = None

        if task == "classification":

            if not is_numeric_dtype(y):

                y = (
                    LabelEncoder()
                    .fit_transform(y)
                )

        else:  # regression

            y_scaler = StandardScaler()

            y = y_scaler.fit_transform(
                y.values.reshape(-1, 1)
            ).flatten()

        # ------------------------
        # Feature Scaling
        # ------------------------

        X_scaler = StandardScaler()

        X = X_scaler.fit_transform(X)

        # ------------------------
        # Train/Test Split
        # ------------------------

        if task == "classification":

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )
            )

        else:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

        # ------------------------
        # Return Everything
        # ------------------------

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            y_scaler
        )