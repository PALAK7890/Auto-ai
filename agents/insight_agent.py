import pandas as pd


class InsightAgent:

    def generate_report(
        self,
        metrics
    ):

        report = []

        report.append(
            "MODEL PERFORMANCE"
        )

        report.append(
            "-" * 30
        )

        for k, v in metrics.items():

            report.append(
                f"{k}: {round(v,4)}"
            )

        return "\n".join(report)