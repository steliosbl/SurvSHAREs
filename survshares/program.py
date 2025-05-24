import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch 

from gplearn.gplearn._program import _Program 

class SurvProgram(_Program): 
    def plot_shape_functions(self, numerical_ranges, categorical_values, steps = 1000):
        shape_arg_ranges, cat_arg_ranges = self.get_argument_ranges_for_shape_functions(numerical_ranges, categorical_values)

        for key, (range_min, range_max) in shape_arg_ranges.items():
            shape = self.model.shape_functions[key]
            t = torch.linspace(range_min, range_max, steps)
            shape.to(torch.device('cpu'))
            with torch.no_grad():
                y = shape(t).flatten()
                plt.plot(t.numpy(),y.numpy())
                plt.show()
        
        for key, (range_min, range_max) in cat_arg_ranges.items():
            range_min, range_max = categorical_values[key]
            shape = self.model.cat_shape_functions[str(key)]
            shape.to(torch.device('cpu'))
            with torch.no_grad():
                t = np.array([range_min, range_max])
                y = shape.numpy() * t
                plt.scatter(t, y)
                plt.show()