import argparse

log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example CLI parser")

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

    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="ERROR",
        help=f"Log levels available: {', '.join(log_levels)}",
    )
    return parser.parse_args()