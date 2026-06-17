import pandas as pd

from agents.dataset_agent import DatasetAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.architecture_agent import ArchitectureAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.insight_agent import InsightAgent

from models.ann import DynamicANN


df = pd.read_csv(
    "datasets/hr.csv"
)

target = "Attrition"


dataset_agent = DatasetAgent()

info = dataset_agent.analyze(df)

task = dataset_agent.detect_task(
    df[target]
)

print(info)

preprocessor = PreprocessingAgent()

X_train, X_test, y_train, y_test = (
    preprocessor.process(
        df,
        target
    )
)

architecture_agent = (
    ArchitectureAgent()
)

hidden_layers = (
    architecture_agent
    .choose_architecture(
        len(df)
    )
)

output_dim = (
    len(df[target].unique())
    if task == "classification"
    else 1
)

model = DynamicANN(
    input_dim=X_train.shape[1],
    hidden_layers=hidden_layers,
    output_dim=output_dim
)

trainer = TrainingAgent()

model = trainer.train(
    model,
    X_train,
    y_train,
    task
)

evaluator = EvaluationAgent()

metrics = evaluator.evaluate(
    model,
    X_test,
    y_test,
    task
)

report = (
    InsightAgent()
    .generate_report(metrics)
)

print(report)