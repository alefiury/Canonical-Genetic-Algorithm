import math
from typing import Tuple, List
from dataclasses import dataclass

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class Colors(metaclass=Singleton):
    GREEN: str = '\033[32m'
    BLUE: str = '\033[34m'
    CYAN: str = '\033[36m'
    WHITE: str = '\033[37m'
    RESET: str = '\033[0m'


@dataclass(frozen=True)
class LogFormatter(metaclass=Singleton):
    colors_single = Colors()
    TIME_DATA: str = colors_single.BLUE + '%(asctime)s' + colors_single.RESET
    MODULE_NAME: str = colors_single.CYAN + '%(module)s' + colors_single.RESET
    LEVEL_NAME: str = colors_single.GREEN + '%(levelname)s' + colors_single.RESET
    MESSAGE: str = colors_single.WHITE + '%(message)s' + colors_single.RESET
    FORMATTER = '['+TIME_DATA+']'+'['+MODULE_NAME+']'+'['+LEVEL_NAME+']'+' - '+MESSAGE


formatter_single = LogFormatter()

def fitness_function(vars):
    x = vars[0]
    y = vars[1]
    return 0.5 - (math.sin(math.sqrt(x**2 + y**2))**2 - 0.5) / (1.0 + 0.001*(x**2 + y**2))**2

def fitness_function_test(x, y):
    return 0.5 - ((math.sin(math.sqrt(x**2 + y**2)**2) - 0.5) / ((1.0 + (0.001*(x**2 + y**2)))**2))


def plot_graph(fitness_function):
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    z = []
    Z = []

    X, Y = np.meshgrid(x, y)

    for i, j in zip(X, Y):
        z = []
        for k, m in zip(i, j):
            z.append(fitness_function_test(k, m))
        Z.append(z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

