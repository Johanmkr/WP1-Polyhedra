import hydra
from omegaconf import DictConfig, OmegaConf
from src_experiment.run_experiment import run
import yaml
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Resolve variables and convert to standard dictionary
    # resolve=True handles the string interpolation for experiment_name
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 2. Save the resolved config to a file for run_experiment.py to read
    # Your current run() function expects a path to a YAML file
    temp_config = "current_run_config.yaml"
    with open(temp_config, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"--- Running experiment: {config_dict['experiment_name']} ---")
    
    # 3. Call your existing experiment logic
    # overwrite=True is safe because Hydra creates unique directories
    run(temp_config, overwrite=True)

if __name__ == "__main__":
    main()