import argparse
import os
import PyADRL.examples.gridworld
from PyADRL.examples.threed import (threed_train)

GRIDWORLD = "gridworld"
THREE_D = "3d"

examples = [GRIDWORLD, THREE_D]


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

    # Example of flag argument:
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print("Verbose mode enabled")

    if args.train:
        if args.train == THREE_D:
            checkpoint = args.checkpoint

            print("Training 3d example:")
            if checkpoint:
                threed_train(
                    checkpoint_path=os.path.abspath(checkpoint)
                )
            else:
                threed_train()
        elif args.train == GRIDWORLD:
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
