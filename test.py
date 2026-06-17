import pandas as pd
from pprint import pprint

from agents.dataset_agent import DatasetAgent

df = pd.read_csv(
    "datasets/HR-Employee-Attrition.csv"
)

agent = DatasetAgent()

analysis = agent.analyze(df)

print("\n=== DATASET ANALYSIS ===\n")

pprint(analysis)

print("\n=== DATASET SUMMARY ===\n")

pprint(
    agent.dataset_summary(df)
)

target = "Attrition"

print("\n=== TASK TYPE ===\n")

print(
    agent.detect_task(
        df[target]
    )
)