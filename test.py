import pandas as pd

from agents.dataset_agent import DatasetAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent

from models.ann import DynamicANN


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "datasets/HR-Employee-Attrition.csv"
)

target = "Attrition"


# ==========================
# Dataset Analysis
# ==========================

dataset_agent = DatasetAgent()

analysis = dataset_agent.analyze(df)

task = dataset_agent.detect_task(
    df[target]
)

print("\n===== DATASET INFO =====")

print("Rows:", analysis["rows"])
print("Columns:", analysis["columns"])

print(
    "Constant Columns:",
    analysis["constant_columns"]
)

print(
    "ID Columns:",
    analysis["id_columns"]
)

print("Task:", task)


# ==========================
# Preprocessing
# ==========================

preprocessor = PreprocessingAgent()

X_train, X_test, y_train, y_test = (
    preprocessor.process(
        df,
        target,
        analysis
    )
)

print("\n===== PREPROCESSING =====")

print(
    "Training Shape:",
    X_train.shape
)

print(
    "Testing Shape:",
    X_test.shape
)


# ==========================
# Build Model
# ==========================

input_dim = X_train.shape[1]

if task == "classification":

    output_dim = len(set(y_train))

else:

    output_dim = 1

model = DynamicANN(
    input_dim=input_dim,
    hidden_layers=[128, 64],
    output_dim=output_dim
)

print(
    "\n===== MODEL CREATED ====="
)

print(
    f"Input Features: {input_dim}"
)

print(
    f"Output Classes: {output_dim}"
)


# ==========================
# Training
# ==========================

trainer = TrainingAgent()

model = trainer.train(
    model,
    X_train,
    y_train,
    epochs=50
)

print(
    "\n===== TRAINING COMPLETE ====="
)


# ==========================
# Evaluation
# ==========================

evaluator = EvaluationAgent()

results = evaluator.evaluate(
    model,
    X_test,
    y_test,
    task
)

print(
    "\n===== EVALUATION RESULTS ====="
)

for metric, value in results.items():

    print(
        f"{metric}: {value:.4f}"
    )