import argparse
from pathlib import Path
import yaml
def get_args():
    parser = argparse.ArgumentParser(
        prog="Experiment Runner",
        description="Run experiments with configurable parameters.",
        epilog="Use a config file or command-line arguments to specify parameters.",
    )
    # Config file argument
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML configuration file. Command-line arguments will override config file values.",
    )
    # Experiment-related values
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--architecture",
        type=int,
        nargs="+",
        default=[3, 3, 3],
        help="Number and layout of hidden layers (e.g., --architecture 3 3 3).",
    )
    # Optimizer-related values
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "RMSprop"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the optimizer.",
    )
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )
    # Parse command-line arguments
    args = parser.parse_args()
    # Load configuration from YAML file if provided
    if args.config:
        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        # Update the default values of the parser with values from the config file
        if "experiment_name" in config_args:
            args.experiment_name = config_args["experiment_name"]
        if "architecture" in config_args:
            args.architecture = config_args["architecture"]
        if "optimizer_args" in config_args:
            if "optimizer" in config_args["optimizer_args"]:
                args.optimizer = config_args["optimizer_args"]["optimizer"]
            if "learning_rate" in config_args["optimizer_args"]:
                args.learning_rate = config_args["optimizer_args"]["learning_rate"]
        if "training_params" in config_args:
            if "epochs" in config_args["training_params"]:
                args.epochs = config_args["training_params"]["epochs"]
    # Validate arguments
    assert args.epochs > 0, "Epochs should be a positive integer."
    assert args.learning_rate > 0, "Learning rate should be a positive float."
    assert len(args.architecture) > 0, "Architecture must have at least one layer."
    return args

def createfolders(*dirs: Path) -> None:
    """
    Create folders for storing data
    """
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    args = get_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
