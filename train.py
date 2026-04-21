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

    # check if target coordinates are within the grid
    if not (0 <= args.target[0] < args.grid[0]) or not (
        0 <= args.target[1] < args.grid[1]
    ):
        raise ValueError(
            f"Target coordinates {args.target} are out of bounds for grid size {args.grid}"
        )

    gridworld_train(
        checkpoint=checkpoint,
        model_name=model_name,
        width=args.grid[0],
        height=args.grid[1],
        target_x=args.target[0],
        target_y=args.target[1],
    )


if __name__ == "__main__":
    main()
