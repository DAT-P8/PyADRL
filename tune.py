from parser import parse_args, log_levels
import os
from PyADRL.examples.gridworld_tuner import gridworld_tune
from PyADRL.utils.logger import Logger
from PyADRL.utils.path_utils import setup_tuner_dir


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

    tuner_dir = setup_tuner_dir()

    gridworld_tune(
        map=args.map,
        tuner_dir=str(tuner_dir),
    )


if __name__ == "__main__":
    main()
