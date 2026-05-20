import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing import Callable, Optional, Any


class Plotter:
    @staticmethod
    def plot_convergence(history: list[float], method: str) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.plot(history, marker="o", label=method)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Root Estimate")
        ax.set_title("Convergence of Root Finding Method")
        ax.legend()
        ax.grid()
        return fig

    @staticmethod
    def plot_function(f: Callable[[float], float], a: float, b: float) -> plt.Figure:
        fig, ax = plt.subplots()
        x = np.linspace(a, b, 100)
        y = f(x)
        ax.plot(x, y)
        ax.set_title("Function Plot")
        ax.grid()
        return fig

    @staticmethod
    def animate_convergence(
        history: list[float],
        errors: list[float],
        method: str,
        interval: int = 500,
    ) -> FuncAnimation:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        def init():
            ax1.set_xlim(0, len(history))
            ax1.set_ylim(min(history) * 0.9, max(history) * 1.1)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Root Estimate")
            ax1.set_title(f"{method} - Root Estimate")
            ax1.grid()

            ax2.set_xlim(0, len(errors))
            ax2.set_ylim(0, max(errors) * 1.1 if errors else 1)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("|f(x)|")
            ax2.set_title(f"{method} - Error")
            ax2.grid()
            ax2.set_yscale("log")
            return ax1, ax2

        def animate(frame):
            ax1.clear()
            ax1.plot(range(frame + 1), history[: frame + 1], marker="o", color="blue")
            ax1.set_xlim(0, len(history))
            ax1.set_ylim(min(history) * 0.9, max(history) * 1.1)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Root Estimate")
            ax1.set_title(f"{method} - Root Estimate")
            ax1.grid()

            ax2.clear()
            ax2.plot(range(frame + 1), errors[: frame + 1], marker="o", color="red")
            ax2.set_xlim(0, len(errors))
            ax2.set_ylim(1e-10, max(errors) * 1.1 if errors else 1)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("|f(x)|")
            ax2.set_title(f"{method} - Error")
            ax2.grid()
            ax2.set_yscale("log")
            return ax1, ax2

        return FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=interval, blit=False)

    @staticmethod
    def live_update_history(ax: plt.Axes, history: list[float], method: str) -> None:
        ax.clear()
        ax.plot(history, marker="o", label=method)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Root Estimate")
        ax.set_title("Convergence (Live)")
        ax.legend()
        ax.grid()
        plt.draw()