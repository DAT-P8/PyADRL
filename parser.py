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
        "--map",
        type=str,
        default="map",
        required=False,
        help="Map name, maps are found in PyADRL/examples/maps. Example: --map map",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="ERROR",
        help=f"Log levels available: {', '.join(log_levels)}",
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        required=False,
        default=0.0,
        help="Delay in seconds between each step.",
    )
    return parser.parse_args()
