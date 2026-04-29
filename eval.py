from PyADRL.utils.logger import Logger
from parser import parse_args, log_levels
import os
from PyADRL.examples.gridworld_eval import gridworld_eval


def main():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    args = parse_args()
    checkpoint = args.restore

    if args.log_level:
        if args.log_level.upper() in log_levels:
            # set env variable to control log level in workers that are spawned by ray
            os.environ["RAY_LOGGER_LEVEL"] = args.log_level.upper()
            Logger.set_level(args.log_level.upper())
        else:
            raise ValueError(
                f"Unknown log level: {args.log_level}. Available log levels: {', '.join(log_levels)}"
            )

    if checkpoint is None:
        print("You need to specify a checkpoint with --restore")
    else:
        print("Evaluating gridworld example:")
        gridworld_eval(
            map=args.map,
            checkpoint_path=checkpoint,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
