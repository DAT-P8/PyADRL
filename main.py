import argparse
import PyADRL.examples.gridworld
import os

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
        "--restore",
        "-r",
        type=str,
        required=False,
        default=None,
        help="Example: ./checkpoint/iter_100 or model_001 (if you have a folder in checkpoints with that name)."
        " If specified, the model will be restored from the checkpoint and training/testing will resume from there.",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=False,
        default=None,
        help="Name for folder that will contain checkpoints inside ./checkpoints. Example: model_001. "
        "Can be used in combination with --restore to train a new model from an old checkpoint",
    )

    parser.add_argument(
        "--grid",
        "-g",
        type=int,
        nargs=2,
        required=False,
        default=[11, 11],
        help="Width and height of the gridworld map (only for gridworld example)",
    )

    parser.add_argument(
        "--target",
        "-t",
        type=int,
        nargs=2,
        required=False,
        default=[5, 5],
        help="X and Y coordinates of the target in the gridworld map (only for gridworld example)",
    )

    return parser.parse_args()


def main():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    args = parse_args()

    if args.train:
        if args.train == GRIDWORLD:
            checkpoint = args.restore
            model_name = args.name if args.name else None

            print("Training gridworld example:")
            PyADRL.examples.gridworld.gridworld_train(
                checkpoint=checkpoint, model_name=model_name
            )
        else:
            print(
                f"Unknown train example: {args.train}. Available examples: {', '.join(examples)}"
            )

    if args.test:
        if args.test == GRIDWORLD:
            checkpoint = args.restore

            if checkpoint is None:
                print("You need to specify a checkpoint with --restore")
            else:
                print("Testing gridworld example:")
                PyADRL.examples.gridworld.gridworld_test(
                    checkpoint_path=checkpoint,
                    width=args.grid[0],
                    height=args.grid[1],
                    target_x=args.target[0],
                    target_y=args.target[1],
                )
        else:
            print(
                f"Unknown test example: {args.test}. Available examples: {', '.join(examples)}"
            )


if __name__ == "__main__":
    main()
