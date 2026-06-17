# test_dataset.py

import pandas as pd

from agents.dataset_agent import DatasetAgent

df = pd.read_csv(
    "datasets/HR-Employee-Attrition.csv"
)

agent = DatasetAgent()

print(agent.analyze(df))