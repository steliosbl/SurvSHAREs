import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from gplearn.gplearn._program import _Program


class SurvProgram(_Program):
    def plot_shape_functions(self, numerical_ranges, categorical_values, steps=1000):
        shape_arg_ranges, cat_arg_ranges = self.get_argument_ranges_for_shape_functions(
            numerical_ranges, categorical_values
        )

        # Line plot data for numerical shape functions
        line_data = []
        for key, (range_min, range_max) in shape_arg_ranges.items():
            shape = self.model.shape_functions[key]
            t = torch.linspace(range_min, range_max, steps)
            shape.to(torch.device("cpu"))
            with torch.no_grad():
                y = shape(t).flatten()
            line_data += [
                dict(x=x, y=y_val, key=str(key))
                for x, y_val in zip(t.numpy(), y.numpy())
            ]

        # Scatter data for categorical shape functions
        scatter_data = []
        for key, (range_min, range_max) in cat_arg_ranges.items():
            range_min, range_max = categorical_values[key]
            shape = self.model.cat_shape_functions[str(key)]
            shape.to(torch.device("cpu"))
            with torch.no_grad():
                t = np.array([range_min, range_max])
                y = shape.numpy() * t
            scatter_data += [
                dict(x=x, y=y_val, key=str(key))
                for x, y_val in zip(t, y)
            ]

        # Plot using seaborn's relplot
        if len(line_data):
            g = sns.relplot(
                data=pd.DataFrame(line_data),
                kind="line",
                x="x",
                y="y",
                col="key",
                col_wrap=3,
                facet_kws=dict(sharex=False, sharey=False),
            )
            g.set_titles(col_template="Numerical: {col_name}")
            plt.show()

        if len(scatter_data):
            g = sns.relplot(
                data=pd.DataFrame(scatter_data),
                kind="scatter",
                x="x",
                y="y",
                col="key",
                col_wrap=3,
                facet_kws=dict(sharex=False, sharey=False),
            )
            g.set_titles(col_template="Categorical: {col_name}")
            plt.show()
