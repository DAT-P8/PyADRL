import argparse
import os
import PyADRL.examples.gridworld

GRIDWORLD = "gridworld"

examples = [GRIDWORLD]


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

    parser.add_argument(
        "--tune",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--testt",
        type=str,
        required=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print("Verbose mode enabled")

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

    if args.tune:
        if args.tune == GRIDWORLD:
            PyADRL.examples.gridworld.gridworld_tuner_train()

    if args.testt:
        PyADRL.examples.gridworld.gridworld_tuner_test(
            checkpoint=os.path.abspath("./checkpoints/iter_/PPO_gridworld_1/checkpoint_000000")
        )


if __name__ == "__main__":
    main()
