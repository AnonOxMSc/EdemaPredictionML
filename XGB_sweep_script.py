# Final sweep running script for XGB hyperparameter tuning using yaml config file
# Compiled by Candidate 1084665


#################################################################
# LOAD REQUIRED PACKAGES

import yaml
import wandb
import os


#################################################################
# MAIN CODE TO BE EXECUTED

for file in os.listdir("FINAL_PIPELINE_AND_TRAINING"):
    if file.endswith(".yaml"):
        yaml_file_path = os.path.join("FINAL_PIPELINE_AND_TRAINING", file)
        with open(yaml_file_path, "r") as yaml_file:
            # Parse the YAML data into a Python dictionary
            sweep_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        sweep_id = wandb.sweep(sweep_config, entity="edema_pred_ml")
        wandb.agent(sweep_id, count=200)


