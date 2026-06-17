import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)


class EvaluationAgent:

    def evaluate(
        self,
        model,
        X_test,
        y_test,
        task,
        y_scaler=None
    ):

        X_test = torch.tensor(
            X_test,
            dtype=torch.float32
        )

        outputs = model(X_test)

        if task == "classification":

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
                .detach()
                .numpy()
            )

            return {
                "accuracy": accuracy_score(
                    y_test,
                    predictions
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                    
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            }

        predictions = (
            outputs.detach()
            .numpy()
            .flatten()
        )

        return {
            "rmse": mean_squared_error(
                y_test,
                predictions
            ) ** 0.5,
            "r2": r2_score(
                y_test,
                predictions
            )
        }