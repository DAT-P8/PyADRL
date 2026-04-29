from parser import parse_args, log_levels
import os
from PyADRL.examples.gridworld_train import gridworld_train
from PyADRL.utils.logger import Logger


def main():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

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

    checkpoint = args.restore
    model_name = args.name if args.name else None

    gridworld_train(args.map, checkpoint=checkpoint, model_name=model_name)


if __name__ == "__main__":
    main()
