import argparse
from datasets import *
from configs import *


def main():
    parser = argparse.ArgumentParser(description="Prepare and visualize dataset statistics")
    parser.add_argument(
        "--config_file", default="configs.yaml", help="path to config file", type=str
    )
    args = parser.parse_args()
    configs_path = os.path.join('configs', args.config_file)
    configs = Configuration(configs_path)
    print(configs)

    datasets = DataPreparing(configs.dataset_path, configs.output_path, configs.ratio_split)
    datasets.create_dataframe()
    datasets.print_dataset_statistics()
    datasets.visualize_dataset_statistics()


if __name__ == '__main__':
    main()
