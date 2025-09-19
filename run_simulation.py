import multiprocessing
import numpy as np
from functools import partial
from training import Training
from environment import MyEnvironment
from fixed_environment import FixedEnvironment
from human import Human
import utils
from configManager import ConfigManager, ConfigConstants

WINDOW_SIZE = utils.WINDOW_SIZE
ALPHA_VECTOR = np.array([0.0,0.01,1.0])

def run_training_with_seed(seed, params=None):
    
    if params == None:
        
        print("No params defined")
        pass
        
    else:    
        """Run a single training process with the specified seed"""
        
        params[ConfigConstants.SEED] = seed
                
        print(f"Starting training with seed {seed}")
            
        #env = MyEnvironment(params)
        env = FixedEnvironment(params)
        human = Human(params)
        
        # Create training instance with the current seed
        trainer = Training(params, env, human)
        
        # Run the training
        trainer.run_training()
        
        print(f"Completed training with seed {seed}")
        
    return trainer
            
if __name__ == "__main__":
            
    params = ConfigManager.load_config("config.yaml")
    ConfigManager.printALL(params)
    
    output_dir = utils.create_output_directories_tree(params[ConfigConstants.NAME_OF_SIM])
    ConfigManager.store_config("config.yaml",output_dir)
    
    # Determine the number of processes to use (use cpu_count or limit to a reasonable number)
    cpu_count = multiprocessing.cpu_count()
    num_processes = min(cpu_count, params[ConfigConstants.NR_OF_SEEDS], multiprocessing.cpu_count())  # Limit to 8 processes max to avoid overwhelming the system
    
    print(f"----- Starting {params[ConfigConstants.NR_OF_SEEDS]} (n of seeds) training sessions with multiprocessing ({num_processes}/{cpu_count} cores available) -----")
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the seeds to the training function
        trainers = pool.map(partial(run_training_with_seed, params=params), range(params[ConfigConstants.NR_OF_SEEDS]))
    
    print("All training sessions completed successfully!")
    
    # Initialize empty 2Darrays for collecting data from all trainers
    all_s_competence = []
    all_c_reward_s = []
    all_eps_s_hist = []
    all_t_actions = []
    all_c_reward_t = []
    all_red_cell_visited = []
    
    # Collect data from all trainers
    for trainer in trainers:
        all_s_competence.append(trainer.student_competence)
        all_c_reward_s.append(trainer.cumulative_reward_s_history)
        all_t_actions.append(trainer.teacher_actions)
        all_c_reward_t.append(trainer.cumulative_reward_t)
        all_red_cell_visited.append(trainer.red_cell_visited_history)
        utils.store_QTable(trainer.QTable,output_dir,seed=trainer.seed)
        
    if True:
        # Convert lists to numpy arrays and calculate MEAN over nr of seeds
        s_competence_mean_over_seeds = np.mean(np.array(all_s_competence), axis=0)
        c_reward_s_mean_over_seeds = np.mean(np.array(all_c_reward_s), axis=0)
        t_actions_history_mean_over_seeds = np.mean(np.array(all_t_actions), axis=0)
        c_reward_t_mean_over_seeds = np.mean(np.array(all_c_reward_t), axis=0)    
        red_cell_visited_over_seeds = np.mean(np.array(all_red_cell_visited), axis=0)
                
        window_size = WINDOW_SIZE        
        weights = np.ones(window_size) / window_size
        
        s_competence_ma = np.convolve(s_competence_mean_over_seeds, weights, mode='valid') 
        c_reward_s_ma   = np.convolve(c_reward_s_mean_over_seeds, weights, mode='valid') 
        teach_act_ma    = np.convolve(t_actions_history_mean_over_seeds, weights, mode='valid') 
        reward_tea_ma   = np.convolve(c_reward_t_mean_over_seeds, weights, mode='valid') 
        red_cell_ma     = np.convolve(red_cell_visited_over_seeds, weights, mode='valid') 
        
        print(f"Plotting results and saving to {output_dir}...")
        
        utils.analyze_data(s_competence_ma,output_dir,f"Student Competence",nr_seeds=params[ConfigConstants.NR_OF_SEEDS])
        utils.analyze_data(c_reward_s_ma,output_dir,f"Student Reward",nr_seeds=params[ConfigConstants.NR_OF_SEEDS])
        utils.analyze_data(teach_act_ma,output_dir,f"Teacher Decisions",nr_seeds=params[ConfigConstants.NR_OF_SEEDS])
        utils.analyze_data(reward_tea_ma,output_dir,f"Teacher Reward",nr_seeds=params[ConfigConstants.NR_OF_SEEDS])
        utils.analyze_data(red_cell_ma,output_dir,f"Red_Cells_Visited",nr_seeds=params[ConfigConstants.NR_OF_SEEDS])

