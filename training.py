from logger import get_logger
from environment import MyEnvironment, MyEnvConstants
from human import Human, HumanConstants
from configManager import ConfigConstants
import numpy as np
import utils

WINDOW_SIZE = utils.WINDOW_SIZE
R_TAU = 'r_tau'
R_MP = 'r_mp'

class Training:
    """Training class for reinforcement learning with teacher-student interaction."""
    
    def __init__(self, params, env: MyEnvironment, human:Human):       
         
        # Init hyperparameters
        self.n_episodes = params[ConfigConstants.N_EPISODES]  # Number of episodes    
        self.gamma_s = params[ConfigConstants.GAMMA_S]  # Discount factor (only for student)
        self.alpha_s = params[ConfigConstants.ALPHA_S] # Learning rate Student
        self.eps_s_init = params[ConfigConstants.EPSILON_S_INIT]
        self.min_eps_s = params[ConfigConstants.MIN_EPSILON_S]
        self.eps_s_mode = params[ConfigConstants.EPS_S_MODE]
        self.eps_s_default = params[ConfigConstants.EPS_S_DEFAULT]
        
        self.alpha_t = params[ConfigConstants.ALPHA_T] # Learning rate Teacher
        self.epsilon_t = params[ConfigConstants.EPSILON_T] # Exploration rate Teacher
        
        self.seed = params[ConfigConstants.SEED]  # seed for reproducibility
        
        self.env = env # environment init
        self.teacher = human # teacher init
        
        self.reward_t = 0.0
        self.reward_s = 0.0
        
        self.student_competence = [] #competence as number of successful/failure episodes (target reached or not)
        self.cumulative_reward_s = 0.0 # cumulative reward over the single episode
        self.cumulative_reward_s_history = [] # cumulative reward over the training for the student
        
        self.cumulative_r_model = [] # Teacher reward model built by the agent
        self.eps_s_history = []
        
        self.teacher_actions = [] # Store the action selection history for the teacher
        self.cumulative_reward_t = [] # cumulative reward over the training for the teacher
        
        self.red_cell_visited_history = []
        
        self.QTable = None
        
        self.count = 0
        
    @staticmethod
    def state_to_index(state, size, dir_max):
        """Convert the state (y, x, d) to a unique index."""
        y, x, d = state
        return (y * size + x) * dir_max + d

    def compute_epsilon_s(self, student_competence, mode):
        """Compute student exploration rate based on competence history."""
        
        if mode == "constant":
            return self.eps_s_default # Default Value
        else:
            eps = self.eps_s_init
            mean_competence = 0
            beta = self.eps_s_init - self.min_eps_s
            if len(student_competence) > 0:
                mean_competence = np.mean(student_competence[- WINDOW_SIZE:])
                eps = self.eps_s_init - beta * mean_competence
            return eps    
    
    def run_training(self):
        
        # Setup logging
        logger = get_logger("training")
        logger.info("Starting training script")

        np.random.seed(self.seed)

        dir_nr = MyEnvConstants.NR_OF_AGENT_DIRECTIONS 
        action_nr = MyEnvConstants.NR_OF_AGENT_ACTIONS 
        states_nr = self.env.height * self.env.width * dir_nr # Calculate the total number of agent states
        self.QTable = np.zeros((states_nr, action_nr)) # Initialize the Q-table with zeros

        Q_Values = {HumanConstants.ACTION_STAY: 0.0, HumanConstants.ACTION_LEAVE: 0.0} # Q-values for human actions

        for ep in range(self.n_episodes):
            logger.info("Starting episode %s/%s", ep+1, self.n_episodes)
            
            # Reset the environment for each episode
            self.env.reset() # (seed=self.seed)
            red_cell_visited = 0
            self.cumulative_reward_s = 0.0
            
            ep_terminated = False
            ep_truncated = False
            # ======================= TEACHER ACTION SELECTION ==================================
            
            # Teacher action selection: to stay or to leave (e-greedy)
            if np.random.uniform(0, 1) < self.epsilon_t:
                t_action = np.random.choice(list(Q_Values.keys()))
                logger.debug("Teacher exploring: selected action '%s' (epsilon=%.2f)", t_action, self.epsilon_t)
            else:
                # Handles the case where Q-values are equal (as at the beginning)
                max_val = max(Q_Values.values())
                max_actions = [a for a, v in Q_Values.items() if v == max_val] #list of actions corresponding to max_val
                t_action = np.random.choice(max_actions)
                logger.debug("Teacher exploiting: selected action '%s' (Q-values: %s)", t_action, Q_Values)
                
            # t_action = "leave"
                
            # ======================= START STUDENT LEARNING PHASE ===============================
                    
            # Select the student exploration rate based on the competence history
            self.epsilon_s = self.compute_epsilon_s(self.student_competence, self.eps_s_mode)

            # Initialize the student state    
            current_state = *self.env.agent_pos, self.env.agent_dir
            current_index = self.state_to_index(current_state, self.env.width, dir_nr)
            
            while not ep_terminated and not ep_truncated:
                self.reward_s = 0.0
                # Check if the student (agent) is on a red cell (teacher preference)
                if self.env._agent_is_on_color_cell(HumanConstants.HUMAN_COLOR_PREF):
                    red_cell_visited += 1        
                
                # Student action selection: go forward, left, right (e-greedy)
                if np.random.uniform(0, 1) < self.epsilon_s:
                    s_action = np.random.randint(0, action_nr)
                else:
                    # Handles the case where Q-values are equal (as at the beginning)
                    max_val = np.max(self.QTable[current_index, :])
                    max_actions = [a for a in range(action_nr) if self.QTable[current_index, a] == max_val]
                    s_action = np.random.choice(max_actions)

                # Take the action and observe the outcome
                new_obs, self.reward_s, ep_terminated, ep_truncated, info = self.env.step(s_action,t_action,self.teacher.r_preference)
                
                next_state = *self.env.agent_pos, self.env.agent_dir
                next_index = self.state_to_index(next_state, self.env.width, dir_nr)
                        
                # Update the Q-value
                self.QTable[current_index, s_action] += self.alpha_s * (self.reward_s + self.gamma_s * np.max(self.QTable[next_index, :]) - self.QTable[current_index, s_action])
                self.cumulative_reward_s += self.reward_s
                
                # Move to the next state
                current_state = next_state
                current_index = next_index    
                                
            # ======================= END STUDENT LEARNING PHASE ==================================
                
            # ======================= UPDATE TEACHER ACTION VALUES  ===============================
                                
            self.reward_t = self.teacher._rewardT(t_action, info[R_TAU], red_cell_visited) # Compute the human reward based on selected action
            Q_Values[t_action] = (1-self.alpha_t) * Q_Values[t_action] + self.alpha_t * self.reward_t # Update the human Q-values

            if ep_terminated == True: # the agent reached the goal (terminal state)
                self.student_competence.append(1)
            else:
                self.student_competence.append(0)
            
            if t_action == HumanConstants.ACTION_STAY:
                self.teacher_actions.append(1)
            else:
                self.teacher_actions.append(0)
            
            self.eps_s_history.append(self.epsilon_s)    
            self.cumulative_reward_s_history.append(self.cumulative_reward_s)
            self.cumulative_reward_t.append(self.reward_t)
            # self.cumulative_r_model.append(info[R_MP])
            
            self.red_cell_visited_history.append(red_cell_visited)
             
        # ======================= END EPISODE ==================================
        
        # Close the environment
        self.env.close()
    