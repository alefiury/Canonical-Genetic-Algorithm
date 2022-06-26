import os
import logging
import argparse

import numpy as np
from omegaconf import OmegaConf

from utils.utils import formatter_single
from utils.genetic_algorithm import GeneticAlgorithm
from utils.utils import fitness_function, plot_graph

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    ag = GeneticAlgorithm(
        **cfg.genetic_algorithm,
        fitness_function=fitness_function
    )

    ag.iterate()


if __name__ == '__main__':
    main()