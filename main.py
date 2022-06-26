import argparse

from omegaconf import OmegaConf

from utils.genetic_algorithm import GeneticAlgorithm
from utils.utils import fitness_function, plot_graph, fitness_function_test

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

    # ag.iterate()

    plot_graph(fitness_function_test)


if __name__ == '__main__':
    main()