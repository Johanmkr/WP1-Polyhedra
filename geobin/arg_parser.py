import argparse
import yaml
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
def parse_and_merge_config():
    """Parse command-line arguments and merge with YAML configuration."""
    # Step 1: Define command-line arguments
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument('--config', type=str, default='../init_file.yml', help="Path to the YAML configuration file")
    parser.add_argument('--experiment_name', type=str, help="Name of the experiment")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs")
    parser.add_argument('--model_type', type=str, help="Type of model (e.g., resnet, vgg)")
    parser.add_argument('--model_layers', type=int, help="Number of layers in the model")
    # Step 2: Parse command-line arguments
    args = parser.parse_args()
    # Step 3: Load configuration from YAML file
    config = load_config(args.config)
    # Step 4: Override YAML settings with command-line arguments (if provided)
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.model_layers:
        config['model']['layers'] = args.model_layers
    # Return the final configuration
    return config
