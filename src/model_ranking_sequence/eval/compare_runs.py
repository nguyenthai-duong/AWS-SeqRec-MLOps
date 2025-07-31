import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from viz import blueq_colors


class ModelMetricsComparisonVisualizer:
    def __init__(
        self,
        curr_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        top_metrics: List[str] = [],
    ):
        self.curr_metrics = curr_metrics
        self.new_metrics = new_metrics

        # Prepare metrics DataFrames
        self.curr_metrics_df = self._prep_metrics_df(curr_metrics)
        self.new_metrics_df = self._prep_metrics_df(new_metrics)
        self.compared_metrics_df = pd.merge(
            self.curr_metrics_df.rename(columns={"value": "curr"}),
            self.new_metrics_df.rename(columns={"value": "new"}),
            on=["metric"],
            how="inner",
            validate="1:1",
        ).assign(
            curr=lambda df: df["curr"],
            diff=lambda df: df["new"] - df["curr"],
            diff_perc=lambda df: df["diff"] / df["curr"],
        )
        self.orderred_list_metrics = top_metrics + [
            metric
            for metric in self.compared_metrics_df.index
            if metric not in top_metrics
        ]
        self.compared_metrics_df = self.compared_metrics_df.loc[
            self.orderred_list_metrics
        ]

    @classmethod
    def _prep_metrics_df(cls, metric_dict: dict):
        df = pd.DataFrame.from_dict(
            metric_dict, orient="index", columns=["value"]
        ).reset_index(names=["metric"])
        selected_metrics = [
            metric for metric in df["metric"].values if metric.startswith("val_")
        ]
        return df.loc[lambda df: df["metric"].isin(selected_metrics)].set_index(
            "metric"
        )

    def compare_metrics_df(self):
        return (
            self.compared_metrics_df.style.format("{:,.2f}")
            .format("{:,.2%}", subset=["diff_perc"])
            .bar(align="mid", color=["red", "green"], subset=["diff_perc"])
        )

    def create_metrics_comparison_plot(self, n_cols=5):
        """
        Create a subplot comparison of metrics between 'curr' and 'new'.

        Parameters:
        - curr_metrics: Data for current metrics
        - new_metrics: Data for new metrics
        - blueq_colors: Object containing color definitions
        - logger: Logger object for logging information
        - n_cols: Number of columns for the subplot grid

        Returns:
        - fig: Plotly figure object
        """

        # Combine DataFrames for processing
        metrics = self.compared_metrics_df.index
        values_curr = self.compared_metrics_df["curr"]
        values_new = self.compared_metrics_df["new"]

        n_rows = int(np.ceil(len(metrics) / n_cols))

        # Create subplots
        fig = make_subplots(rows=n_rows, cols=n_cols)

        # Add data for each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols + 1
            col = i % n_cols + 1

            # Add trace for 'curr'
            fig.add_trace(
                go.Bar(
                    name="curr",
                    x=[metric],
                    y=[values_curr[metric]],
                    marker_color=blueq_colors.main,
                    showlegend=(i == 0),
                    texttemplate="%{y:.2}",
                ),
                row=row,
                col=col,
            )

            # Add trace for 'new'
            fig.add_trace(
                go.Bar(
                    name="new",
                    x=[metric],
                    y=[values_new[metric]],
                    marker_color=blueq_colors.others[0],
                    showlegend=(i == 0),
                    texttemplate="%{y:.2}",
                ),
                row=row,
                col=col,
            )

            # Add diff annotation
            difference = (values_new[metric] - values_curr[metric]) / values_curr[
                metric
            ]
            fig.add_annotation(
                x=metric,
                y=values_curr[metric] * 1.10,  # Position above the tallest bar
                text=f"Δ={difference:.2%}",
                showarrow=False,
                font=dict(color="black", size=14),
                row=row,
                col=col,
            )

        # Update layout
        fig.update_layout(
            height=400 * n_rows,
            title="Metric Comparisons",
            showlegend=True,
            bargroupgap=0.3,
            bargap=0.5,
        )

        # Hide y-axes
        for axis in fig.layout:
            if axis.startswith("yaxis"):
                fig.layout[axis].visible = False  # Hide y-axis labels

        fig.show()

    def plot_diff(self):
        compared_metrics_df = self.compared_metrics_df.reset_index()

        # Bar plot
        fig = px.bar(
            compared_metrics_df,
            x="metric",
            y=["diff_perc"],
            barmode="group",
        )
        fig.update_layout(
            title="Metric Diff Comparisons",
            xaxis_title="Metric",
            yaxis_title="% Difference (new - curr)",
        )

        fig.show()
