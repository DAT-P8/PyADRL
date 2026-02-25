import argparse
import PyADRL.examples.gridworld;

GRIDWORLD = "gridworld"

examples = [GRIDWORLD]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example CLI parser"
    )

    parser.add_argument(
        "--example",
        type=str,
        required=False,
        help=f"Examples available: {", ".join(examples)}"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Only for quick testing"
    )

    # Example of number parsing:
    # parser.add_argument(
    #     "--number",
    #     type=int,
    #     required=False,
    #     help="An optional integer argument"
    # )

    # Example of flag argument:
    # parser.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     help="Enable verbose output"
    # )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.test:
        PyADRL.examples.gridworld.py_torch_tutorial()
        return;

    if args.verbose:
        print("Verbose mode enabled")

    if args.example:
        if (args.example == GRIDWORLD):
            print("Running gridworld example:")
            PyADRL.examples.gridworld.gridworld_example()

    if args.number is not None:
        print(f"Number argument: {args.number}")


if __name__ == "__main__":
    main()
