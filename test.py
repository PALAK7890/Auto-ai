import pandas as pd

from agents.dataset_agent import DatasetAgent
from agents.preprocessing_agent import (
    PreprocessingAgent
)

df = pd.read_csv(
    "datasets/HR-Employee-Attrition.csv"
)

dataset_agent = DatasetAgent()

analysis = dataset_agent.analyze(df)

processor = PreprocessingAgent()

X_train, X_test, y_train, y_test = (
    processor.process(
        df,
        target="Attrition",
        analysis=analysis
    )
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("Train Labels:", len(y_train))
print("Test Labels :", len(y_test))