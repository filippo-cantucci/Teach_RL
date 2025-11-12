import multiprocessing
import numpy as np
from functools import partial
from training import Training
from environment import MyEnvironment
from human import Human
import utils
from configManager import ConfigManager


def run_training_over_multiple_envs(seed, environments=None, params=None):

    leverage_nr_of_ep_for_learning_policy = 7000
    nr_of_ep_for_env_change = 1000
    
    n = leverage_nr_of_ep_for_learning_policy // nr_of_ep_for_env_change
    
    env = MyEnvironment(params)
    human = Human(params)
    trainer = Training(params, env, human)

    for i in range(1,n+1):
        for j in range(len(environments)):
            params[ConfigManager.LAYOUT_V] = environments[j]
            env.rebuild_env(env.width,env.height)
            trainer.set_environment(env)
            trainer.run_training()
            
            print(f"Completed training with environment {environments[j]}")
            
    return trainer


def run_training_with_seed(seed, params=None):
    
    if params == None:
        
        print("No params defined")
        pass
        
    else:    
        """Run a single training process with the specified seed"""
        
        params[ConfigManager.SEED] = seed
                
        print(f"Starting training with seed {seed}")
            
        #env = MyEnvironment(params)
        env = MyEnvironment(params)
        human = Human(params)
        
        # Create training instance with the current seed
        trainer = Training(params, env, human)
        
        # Run the training
        trainer.run_training()
        
        print(f"Completed training with seed {seed}")
        
    return trainer
            
if __name__ == "__main__":
            
    params = ConfigManager.load_config("config.yaml")
    ConfigManager().printALL(params)
    
    output_dir = utils.create_output_directories_tree(params[ConfigManager.NAME_OF_SIM])
    ConfigManager.store_config("config.yaml",output_dir)
    
    environments = ["v1","v2","v3","v4"]
    
    # Determine the number of processes to use (use cpu_count or limit to a reasonable number)
    cpu_count = multiprocessing.cpu_count()
    num_processes = min(cpu_count, params[ConfigManager.NR_OF_SEEDS], multiprocessing.cpu_count())  # Limit to 8 processes max to avoid overwhelming the system
    print(f"----- Starting {params[ConfigManager.NR_OF_SEEDS]} (n of seeds) training sessions with multiprocessing ({num_processes}/{cpu_count} cores available) -----")
    
    mode = params[ConfigManager.SIM_MODE]
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        
        if mode == "single_env":
            map_fn = partial(run_training_with_seed, params=params)
        elif mode == "multiple_env":
            map_fn = partial(run_training_over_multiple_envs, environments=environments, params=params)
        
        # Map the seeds to the training function
        trainers = pool.map(map_fn, range(params[ConfigManager.NR_OF_SEEDS]))
    
    print("All training sessions completed successfully!")
    
    # Initialize empty 2Darrays for collecting data from all trainers
    all_s_competence = []
    all_c_reward_s = []
    all_eps_s_hist = []
    all_t_actions = []
    all_c_reward_t = []
    
    # Collect data from all trainers
    for trainer in trainers:
        all_s_competence.append(trainer.student_competence)
        all_c_reward_s.append(trainer.cumulative_reward_s_trend)
        all_t_actions.append(trainer.cumulative_teacher_actions)
        all_c_reward_t.append(trainer.cumulative_reward_teacher)
        
    if True:
        # Convert lists to numpy arrays and calculate MEAN over nr of seeds
        s_competence_mean_over_seeds = np.mean(np.array(all_s_competence), axis=0)
        c_reward_s_mean_over_seeds = np.mean(np.array(all_c_reward_s), axis=0)
        t_actions_history_mean_over_seeds = np.mean(np.array(all_t_actions), axis=0)
        c_reward_t_mean_over_seeds = np.mean(np.array(all_c_reward_t), axis=0)    
                
        weights = np.ones(utils.WINDOW_SIZE) / utils.WINDOW_SIZE
        
        s_competence_ma = np.convolve(s_competence_mean_over_seeds, weights, mode='valid') 
        c_reward_s_ma   = np.convolve(c_reward_s_mean_over_seeds, weights, mode='valid') 
        teach_act_ma    = np.convolve(t_actions_history_mean_over_seeds, weights, mode='valid') 
        reward_tea_ma   = np.convolve(c_reward_t_mean_over_seeds, weights, mode='valid') 
        
        print(f"Plotting results and saving to {output_dir}...")
        
        utils.analyze_data(s_competence_ma,output_dir,f"Student Competence",nr_seeds=params[ConfigManager.NR_OF_SEEDS])
        utils.analyze_data(c_reward_s_ma,output_dir,f"Student Reward",nr_seeds=params[ConfigManager.NR_OF_SEEDS])
        utils.analyze_data(teach_act_ma,output_dir,f"Teacher Decisions",nr_seeds=params[ConfigManager.NR_OF_SEEDS])
        utils.analyze_data(reward_tea_ma,output_dir,f"Teacher Reward",nr_seeds=params[ConfigManager.NR_OF_SEEDS])

