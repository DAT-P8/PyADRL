import argparse
import os
import PyADRL.examples.gridworld

GRIDWORLD = "gridworld"

examples = [GRIDWORLD]
log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example CLI parser")

    parser.add_argument(
        "--train",
        type=str,
        required=False,
        help=f"Examples available: {', '.join(examples)}",
    )

    parser.add_argument(
        "--test",
        type=str,
        required=False,
        help=f"Examples available: {', '.join(examples)}",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Example: ./checkpoint/iter_100",
    )

    # Example of number parsing:
    parser.add_argument(
        "--number", type=int, required=False, help="An optional integer argument"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        help=f"Log levels available: {', '.join(log_levels)}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_level:
        if args.log_level.upper() in log_levels:
            os.environ["RAY_LOGGER_LEVEL"] = args.log_level.upper()
    else:
        os.environ["RAY_LOGGER_LEVEL"] = "ERROR"

    if args.train:
        if args.train == GRIDWORLD:
            checkpoint = args.checkpoint

            print("Training gridworld example:")
            if checkpoint:
                PyADRL.examples.gridworld.gridworld_train(
                    checkpoint_path=os.path.abspath(checkpoint)
                )
            else:
                PyADRL.examples.gridworld.gridworld_train()

    if args.test:
        if args.test == GRIDWORLD:
            checkpoint = args.checkpoint

            if checkpoint is None:
                print("You need to specify a checkpoint with --checkpoint")
            else:
                print("Testing gridworld example:")
                PyADRL.examples.gridworld.gridworld_test(
                    checkpoint_path=os.path.abspath(checkpoint)
                )

    if args.number is not None:
        print(f"Number argument: {args.number}")


if __name__ == "__main__":
    main()
