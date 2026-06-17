import streamlit as st
import pandas as pd

from agents.dataset_agent import DatasetAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from models.ann import DynamicANN


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AutoInsight AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoInsight AI")
st.markdown(
    "Upload any tabular dataset, train a PyTorch neural network, and analyze performance."
)

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # ==========================
        # DATASET PREVIEW
        # ==========================

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        # ==========================
        # TARGET COLUMN
        # ==========================

        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        # ==========================
        # DATASET ANALYSIS
        # ==========================

        dataset_agent = DatasetAgent()

        analysis = dataset_agent.analyze(df)

        task = dataset_agent.detect_task(
            df[target]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                analysis["rows"]
            )

        with col2:
            st.metric(
                "Columns",
                analysis["columns"]
            )

        with col3:
            st.metric(
                "Task",
                task
            )

        with st.expander(
            "Dataset Analysis"
        ):

            st.json(analysis)

        # ==========================
        # TRAIN MODEL BUTTON
        # ==========================

        if st.button(
            "🚀 Train Model"
        ):

            with st.spinner(
                "Training model..."
            ):

                # --------------------
                # Preprocessing
                # --------------------

                preprocessor = (
                    PreprocessingAgent()
                )

                X_train, X_test, y_train, y_test,y_scaler = (
                    preprocessor.process(
                        df,
                        target,
                        analysis,
                        task
                    )
                )

                # --------------------
                # Model Creation
                # --------------------

                input_dim = (
                    X_train.shape[1]
                )

                if task == "classification":

                    output_dim = len(
                        set(y_train)
                    )

                else:

                    output_dim = 1

                model = DynamicANN(
                    input_dim=input_dim,
                    hidden_layers=[128, 64],
                    output_dim=output_dim
                )

                # --------------------
                # Training
                # --------------------

                trainer = TrainingAgent()

                

                model = trainer.train(
                        model,
                        X_train,
                        y_train,
                        task,
                        epochs=50
                    )

                

                # --------------------
                # Evaluation
                # --------------------

                evaluator = (
                    EvaluationAgent()
                )

                metrics = (
                    evaluator.evaluate(
                        model,
                        X_test,
                        y_test,
                        task,
                        y_scaler
                    )
                )

            st.success(
                "Training Complete!"
            )

            # ==========================
            # METRICS
            # ==========================

            st.subheader(
                "Model Performance"
            )

            metric_cols = st.columns(
                len(metrics)
            )

            for idx, (
                metric,
                value
            ) in enumerate(
                metrics.items()
            ):

                if isinstance(
                    value,
                    (int, float)
                ):

                    metric_cols[idx].metric(
                        metric.upper(),
                        f"{value:.4f}"
                    )

            # ==========================
            # RAW RESULTS
            # ==========================

            with st.expander(
                "Raw Metrics"
            ):

                st.json(metrics)

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Upload a CSV file to begin."
    )