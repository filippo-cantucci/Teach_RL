import yaml
import copy
import os
import shutil
from typing import Dict, Any, Optional
from enum import Enum

class ConfigManager():
    
    NAME_OF_SIM = "name_of_sim"
    NR_OF_SEEDS = "nr_of_seeds"
    MAX_STEPS = "max_steps"
    N_EPISODES = "n_episodes"
    SEED = "seed"
    GRID_SIZE = "grid_size"
    RENDER_MODE ="render_mode"
    ALPHA_S = "alpha_s"
    EPSILON_S_INIT = "epsilon_s_init"
    MIN_EPSILON_S = "min_epsilon_s"
    GAMMA_S = "gamma_s"
    ALPHA_REW_MODEL = "alpha_rew_model"
    ALPHA_T = "alpha_t"
    EPSILON_T = "epsilon_t"
    EPS_S_MODE = "eps_s_mode"
    EPS_S_DEFAULT = "eps_s_default"
    HUMAN_PREFERENCES = "human_preferences"
    ABSENCE_MUX = "absence_multiplier"
    LAYOUT_V = "layout_version"
    
    @staticmethod
    def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  
        
        try:
            with open(config_path, "r") as f:
                params = yaml.safe_load(f)
        except FileNotFoundError:
            raise
        except yaml.YAMLError as e:
            raise
        
        # create a copy without modifying the config.yaml
        params = copy.deepcopy(params)
        
        if overrides:
            for k, v in overrides.items():
                params[k] = v
            
        return params
    
    @staticmethod
    def store_config(config_file_path: str, folder_path: str, filename: str = "config.yaml"):

        try: 
            # Create full path to destination config file
            destination_path = os.path.join(folder_path, filename)
            
            # Copy the config file to the destination
            shutil.copy2(config_file_path, destination_path)
                        
        except FileNotFoundError:
            raise
        except OSError as e:
            raise
    
    def printALL(self,params):
        
        print("\n" + "="*50)
        print("CONFIGURATION PARAMETERS")
        print("="*50)
        
        # Simulation parameters
        print("\n SIMULATION:")
        print(f"  Name of simulation: {params.get(self.NAME_OF_SIM, 'N/A')}")
        print(f"  Number of seeds: {params.get(self.NR_OF_SEEDS, 'N/A')}")
        print(f"  Max steps: {params.get(self.MAX_STEPS, 'N/A')}")
        print(f"  Number of episodes: {params.get(self.N_EPISODES, 'N/A')}")
        print(f"  Seed: {params.get(self.SEED, 'N/A')}")
        
        # Environment parameters
        print("\n  ENVIRONMENT:")
        print(f"  Grid size: {params.get(self.GRID_SIZE, 'N/A')}")
        print(f"  Render mode: {params.get(self.RENDER_MODE, 'N/A')}")
        print(f"  Environment Layout: {params.get(self.LAYOUT_V, 'N/A')}")
        
        # Student parameters
        print("\n STUDENT HYPERPARAMETERS:")
        print(f"  Alpha (learning rate): {params.get(self.ALPHA_S, 'N/A')}")
        print(f"  Epsilon init: {params.get(self.EPSILON_S_INIT, 'N/A')}")
        print(f"  Min epsilon: {params.get(self.MIN_EPSILON_S, 'N/A')}")
        print(f"  Gamma (discount): {params.get(self.GAMMA_S, 'N/A')}")
        print(f"  Alpha reward model: {params.get(self.ALPHA_REW_MODEL, 'N/A')}")
        print(f"  Epsilon mode: {params.get(self.EPS_S_MODE, 'N/A')}")
        print(f"  Epsilon default: {params.get(self.EPS_S_DEFAULT, 'N/A')}")
        
        # Teacher parameters
        print("\n TEACHER HYPERPARAMETERS:")
        print(f"  Alpha (learning rate): {params.get(self.ALPHA_T, 'N/A')}")
        print(f"  Epsilon: {params.get(self.EPSILON_T, 'N/A')}")
        
        print("="*50 + "\n")
        
        while True:
            response = input("Do you want to start simulation? (y/n): ").lower().strip()
            if response in ['y','yes']:
                break
            elif response in ['n','no']:
                print("Simulation killed by the user")
                exit(0)
            else:
                print("Please answer 'y' or 'n'")
                
        
        
    
    