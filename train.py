from parser import parse_args, log_levels
import os
from PyADRL.examples.gridworld_train import gridworld_train
from PyADRL.utils.logger import Logger
from PyADRL.utils.path_utils import setup_model_dir


def main():
    args = parse_args()

    if args.log_level:
        if args.log_level.upper() in log_levels:
            # set env variable to control log level in workers that are spawned by ray
            os.environ["RAY_LOGGER_LEVEL"] = args.log_level.upper()
            Logger.set_level(args.log_level.upper())
        else:
            raise ValueError(
                f"Unknown log level: {args.log_level}. Available log levels: {', '.join(log_levels)}"
            )

    gridworld_train(args.map, training_path=setup_model_dir())


if __name__ == "__main__":
    main()
