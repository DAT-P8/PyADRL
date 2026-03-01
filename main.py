import argparse
import os
import PyADRL.examples.gridworld;

GRIDWORLD = "gridworld"

examples = [GRIDWORLD]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example CLI parser"
    )

    parser.add_argument(
        "--train",
        type=str,
        required=False,
        help=f"Examples available: {", ".join(examples)}"
    )

    parser.add_argument(
        "--test",
        type=str,
        required=False,
        help=f"Examples available: {", ".join(examples)}"
    )

    # Example of number parsing:
    parser.add_argument(
        "--number",
        type=int,
        required=False,
        help="An optional integer argument"
    )

    # Example of flag argument:
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print("Verbose mode enabled")

    if args.train:
        if (args.train == GRIDWORLD):
            print("Training gridworld example:")
            PyADRL.examples.gridworld.gridworld_train()

    if args.test:
        if (args.test == GRIDWORLD):
            print("Testing gridworld example:")
            PyADRL.examples.gridworld.gridworld_test(checkpoint_path=os.path.abspath("./checkpoints/iter_100"))

    if args.number is not None:
        print(f"Number argument: {args.number}")


if __name__ == "__main__":
    main()
