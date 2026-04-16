import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    @staticmethod
    def plot_convergence(history, method):
        plt.plot(history, marker = 'o', label = method)
        plt.xlabel("Iteration")
        plt.ylabel("Root Estimate")
        plt.title("Convergence of Root Finding Method")
        plt.legend()
        plt.grid()
        # plt.show(/)
    

    @staticmethod
    def plot_function(f, a, b):
        x = np.linspace(a, b, 100)
        y = f(x)
        plt.plot(x, y)
        plt.title("Function Plot")
        plt.grid()
        plt.show()