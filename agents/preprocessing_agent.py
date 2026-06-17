from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class PreprocessingAgent:

    def process(self, df, target):

        X = df.drop(columns=[target])
        y = df[target]

        for col in X.columns:

            if X[col].dtype == "object":

                encoder = LabelEncoder()

                X[col] = encoder.fit_transform(
                    X[col].astype(str)
                )

        if y.dtype == "object":

            target_encoder = LabelEncoder()

            y = target_encoder.fit_transform(y)

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )