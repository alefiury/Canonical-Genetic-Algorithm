import argparse

from omegaconf import OmegaConf

from utils.genetic_algorithm import GeneticAlgorithm
from utils.utils import fitness_function, plot_graph

def main() -> None:
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

    ga = GeneticAlgorithm(
        **cfg.genetic_algorithm,
        fitness_function=fitness_function
    )

    ga.iterate()
    # plot_graph()

if __name__ == '__main__':
    main()