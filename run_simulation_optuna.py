import multiprocessing
import numpy as np
import optuna
import copy
from functools import partial
from typing import Dict, Any, List, Optional

from training import Training
from environment import MyEnvironment
from human import Human
import utils
from logger import setup_logging, get_logger
from configManager import ConfigManager, ConfigConstants

# Setup logging and get logger for this module
setup_logging()
logger = get_logger(__name__)

# Configuration
WINDOW_SIZE = 20
MAX_PROCESSES = 8
ALPHA_VECTOR = np.arange(0.0, 1.5, 0.5)
NR_OF_SEEDS = 30
PROVA = "PROVISSIMA"


def run_training_with_seed(seed: int, params: Dict[str, Any]) -> Training:
    
    """Run a single training process with the specified seed"""
    # Create a copy of params to avoid modifying the original
    params_copy = copy.deepcopy(params)
    params_copy[ConfigConstants.SEED] = seed
    
    logger.info(f"Starting training with seed {seed}")
        
    env = MyEnvironment(params=params_copy)
    human = Human(params=params_copy)
    
    # Create training instance with the current seed
    trainer = Training(params=params_copy, env=env, human=human)
    
    # Run the training
    trainer.run_training()
    
    logger.info(f"Completed training with seed {seed}")
    
    return trainer

def process_and_save_results(final_results: List[Training], params: Dict[str, Any]) -> float:
    
    """Process training results and save plots for a specific alpha_rew_model"""
    # Initialize empty 2D arrays for collecting data from all trainers
    all_s_competence = []
    all_c_reward_s = []
    all_c_r_model = []
    all_eps_s_hist = []
    all_t_actions = []
    all_c_reward_t = []
    
    # Collect data from all trainers
    for trainer in final_results:
        all_s_competence.append(trainer.student_competence)
        all_c_reward_s.append(trainer.cumulative_reward_s)
        all_c_r_model.append(trainer.cumulative_r_model)
        all_eps_s_hist.append(trainer.eps_s_history)
        all_t_actions.append(trainer.teacher_actions)
        all_c_reward_t.append(trainer.cumulative_reward_t)

    # Convert lists to numpy arrays and calculate MEAN over nr of seeds
    s_competence_mean_over_seeds = np.mean(np.array(all_s_competence), axis=0)
    c_reward_s_mean_over_seeds = np.mean(np.array(all_c_reward_s), axis=0)
    c_r_model_mean_over_seeds = np.mean(np.array(all_c_r_model), axis=0)
    eps_s_hist_mean_over_seeds = np.mean(np.array(all_eps_s_hist), axis=0)
    t_actions_history_mean_over_seeds = np.mean(np.array(all_t_actions), axis=0)
    c_reward_t_mean_over_seeds = np.mean(np.array(all_c_reward_t), axis=0)    
            
    window_size = WINDOW_SIZE
    logger.info(f"Calculating moving averages with window size {window_size}...")
    
    weights = np.ones(window_size) / window_size
    
    s_competence_ma = np.convolve(s_competence_mean_over_seeds, weights, mode='valid')
    c_reward_s_ma   = np.convolve(c_reward_s_mean_over_seeds, weights, mode='valid')
    c_r_model_ma    = np.convolve(c_r_model_mean_over_seeds, weights, mode='valid')
    eps_s_hist_ma   = np.convolve(eps_s_hist_mean_over_seeds, weights, mode='valid')
    teach_act_ma    = np.convolve(t_actions_history_mean_over_seeds, weights, mode='valid')
    reward_tea_ma   = np.convolve(c_reward_t_mean_over_seeds, weights, mode='valid')
    
    output_dir = utils.create_output_directories_tree(params[ConfigConstants.ALPHA_REW_MODEL])
    
    logger.info(f"Plotting results and saving to {output_dir}...")
    
    utils.plot_vector(s_competence_ma, f"Student Competence_{params[ConfigConstants.ALPHA_REW_MODEL]}")
    utils.plot_vector(c_reward_s_ma, f"Student Reward_{params[ConfigConstants.ALPHA_REW_MODEL]}")
    utils.plot_vector(c_r_model_ma, f"Teacher Preferences Model_{params[ConfigConstants.ALPHA_REW_MODEL]}")
    utils.plot_vector(eps_s_hist_ma, f"Epsilon student_{params[ConfigConstants.ALPHA_REW_MODEL]}")
    utils.plot_vector(teach_act_ma, f"Teacher Decisions_{params[ConfigConstants.ALPHA_REW_MODEL]}")
    utils.plot_vector(reward_tea_ma, f"Teacher Reward_{params[ConfigConstants.ALPHA_REW_MODEL]}")
    
    # Return a metric that could be used for optimization (e.g., final student competence)
    return s_competence_ma[-1] if len(s_competence_ma) > 0 else 0.0

def objective(trial: optuna.Trial) -> float:
    
    """Optuna objective function"""
    # Suggest alpha_rew_model value from the discrete set
    alpha_rew_model = trial.suggest_float(ConfigConstants.ALPHA_REW_MODEL, ALPHA_VECTOR)
    
    logger.info(f"\n=== Running simulations for alpha_rew_model = {alpha_rew_model} ===")
    
    # Create parameters with overrides for this trial
    params_to_override = {ConfigConstants.NR_OF_SEEDS: NR_OF_SEEDS, 
                         ConfigConstants.ALPHA_REW_MODEL: alpha_rew_model}
    
    params = ConfigManager.load_config("config.yaml", params_to_override)
    
    # Number of different seeds to use
    logger.info(f"Starting {params_to_override[ConfigConstants.NR_OF_SEEDS]} training sessions with multiprocessing")
    
    # Determine the number of processes to use
    cpu_count = multiprocessing.cpu_count()
    num_processes = min(cpu_count, params_to_override[ConfigConstants.NR_OF_SEEDS], MAX_PROCESSES)  # Limit to MAX_PROCESSES max
    
    logger.info(f"Using {num_processes} processes on {cpu_count} available CPU cores")
    
    # Create a pool of workers
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the seeds to the training function
            final_results = pool.map(partial(run_training_with_seed, params=params), range(params_to_override[ConfigConstants.NR_OF_SEEDS]))
    except Exception as e:
        logger.error(f"Error during multiprocessing: {e}")
        raise
    
    logger.info(f"All training sessions completed for alpha_rew_model = {alpha_rew_model}!")
    
    # Process and save results for this alpha value and return the metric
    logger.info(f"Alpha rew model: {alpha_rew_model}")
    final_metric = process_and_save_results(final_results, params=params)
    
    return final_metric
            
            
if __name__ == "__main__":
    try:
        logger.info("Starting parameter sweep for alpha_rew_model values from 0.0 to 1.0 with step 0.2")
        
        # Define all values to test
        alpha_values = ALPHA_VECTOR
        
        # Create Optuna study for parameter exploration
        study = optuna.create_study(
            study_name="alpha_rew_model_sweep",  # New study name to ensure fresh start
            direction="maximize",  # Maximize final student competence
            storage="sqlite:///optuna_study.db",  # Persist study results
            load_if_exists=False  # Always create new study
        )
        
        # Enqueue all specific values we want to test
        for alpha_value in alpha_values:
            study.enqueue_trial({ConfigConstants.ALPHA_REW_MODEL: alpha_value})
        
        # Run optimization - this will test all enqueued values first
        study.optimize(objective, n_trials=len(alpha_values))
        
        logger.info("\n=== All parameter sweep simulations completed! ===")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best value (final student competence): {study.best_value}")
        
        # Print all trials results
        logger.info("\nAll trials results:")
        for trial in study.trials:
            logger.info(f"Alpha: {trial.params[ConfigConstants.ALPHA_REW_MODEL]}, Final Competence: {trial.value}")
        
        logger.info("Results saved for all alpha_rew_model values from 0.0 to 1.0")
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise