from parser import parse_args, log_levels
import os
import PyADRL.examples.gridworld

def main():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    args = parse_args()
    checkpoint = args.restore

    if checkpoint is None:
        print("You need to specify a checkpoint with --restore")
    else:
        print("Evaluating gridworld example:")
        PyADRL.examples.gridworld.gridworld_test(
            checkpoint_path=checkpoint,
            width=args.grid[0],
            height=args.grid[1],
            target_x=args.target[0],
            target_y=args.target[1],
        )

    if args.log_level:
        if args.log_level.upper() in log_levels:
            os.environ["RAY_LOGGER_LEVEL"] = args.log_level.upper()
        else:
            raise ValueError(f"Unknown log level: {args.log_level}. Available log levels: {', '.join(log_levels)}")
    
    checkpoint = args.restore
    model_name = args.name if args.name else None

    print("Training gridworld example:")
    PyADRL.examples.gridworld.gridworld_train(
        checkpoint=checkpoint, model_name=model_name
    )

if __name__ == "__main__":
    main()