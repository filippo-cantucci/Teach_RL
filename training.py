from environment import MyEnvironment
from human import Human
from configManager import ConfigManager
import numpy as np
import utils

class Training:
    
    """Training class for reinforcement learning with teacher-student interaction."""
    
    def __init__(self, params, env: MyEnvironment, human:Human):
        
        print("Training INIT ")       
         
        self.cfg = params           
        self.env = env # environment init
        self.teacher = human # teacher init
        
        self.student_competence = [] #competence as number of successful/failure episodes (target reached or not) (per tutti gli episodi)
        self.cumulative_reward_s_trend = [] # cumulative reward trend over the training for the student  (per tutti gli episodi)
        
        self.cumulative_teacher_actions = []  # cumulative actions selected over the training for the teacher (per tutti gli episodi)
        self.cumulative_reward_teacher = []   # cumulative reward over the training for the teacher (per tutti gli episodi)
        
        self.student_QTable_Dict = {k: np.zeros((env.height * env.width * env.NR_OF_ROBOT_DIRECTIONS,env.NR_OF_ROBOT_ACTIONS)) for k in ("v1","v2","v3","v4")}
        self.teacher_Q_Values = {human.HUMAN_ACTION_STAY: 0.0, human.HUMAN_ACTION_LEAVE: 0.0} # Q-values for human actions
        
    @staticmethod
    def state_to_index(state, size, dir_max):
        """Convert the state (y, x, d) to a unique index."""
        y, x, d = state
        return (y * size + x) * dir_max + d

    def compute_epsilon_s(self, student_competence, mode):
        
        """Compute student exploration rate based on competence history."""
        eps_init = self.cfg[ConfigManager.EPS_S_DEFAULT] # Default Value
        
        if mode == "constant":
            return eps_init
        else:
            eps = eps_init
            mean_competence = 0
            beta = eps_init - self.cfg[ConfigManager.MIN_EPSILON_S]
            if len(student_competence) > 0:
                mean_competence = np.mean(student_competence[- utils.WINDOW_SIZE:])
                eps = eps_init - beta * mean_competence
            return eps    
            
    def reset_training(self):
        np.random.seed(self.cfg[ConfigManager.SEED])
        self.student_competence = []
        self.cumulative_reward_s_trend = []
        self.cumulative_teacher_actions = []  
        self.cumulative_reward_teacher = []
            
    def set_environment(self,env):
        self.env = env
         
    def set_human_teacher(self,human):
        self.teacher = human
    
    def run_training(self):
        current_student_QTable = self.student_QTable_Dict[self.cfg[ConfigManager.LAYOUT_V]]
        if self.cfg[ConfigManager.SIM_MODE] == "single_env":
            episodes = self.cfg[ConfigManager.N_EPISODES_SINGLE_ENV]
        elif self.cfg[ConfigManager.SIM_MODE] == "multiple_env":
            episodes = self.cfg[ConfigManager.N_EPISODES_MULTIPLE_ENV]
                        
        for ep in range(episodes):   
            self.env.reset(self.cfg[ConfigManager.SEED])
            cumulative_reward_s = 0.0  # cumulative reward over a single episode for the student (su tutto l'episodio)
            ep_terminated = False
            ep_truncated = False
            
            # ======================= TEACHER ACTION SELECTION ==================================
            
            # Teacher action selection: to stay or to leave (e-greedy)
            if np.random.uniform(0, 1) < self.cfg[ConfigManager.EPSILON_T]:
                t_action = np.random.choice(list(self.teacher_Q_Values.keys()))
            else:
                # Handles the case where Q-values are equal (as at the beginning)
                max_val = max(self.teacher_Q_Values.values())
                max_actions = [a for a, v in self.teacher_Q_Values.items() if v == max_val] #list of actions corresponding to max_val
                t_action = np.random.choice(max_actions)
                
            t_action = self.teacher.HUMAN_ACTION_STAY
                
            # ======================= START STUDENT LEARNING PHASE ===============================
                    
            # Select the student exploration rate based on the competence history
            self.epsilon_s = self.compute_epsilon_s(self.student_competence, self.cfg[ConfigManager.EPS_S_MODE])

            # Initialize the student state    
            current_state = *self.env.agent_pos, self.env.agent_dir
            current_index = self.state_to_index(current_state, self.env.width, self.env.NR_OF_ROBOT_DIRECTIONS)
                        
            while not ep_terminated and not ep_truncated:   
                                                    
                # Student action selection: go forward, left, right (e-greedy)
                if np.random.uniform(0, 1) < self.epsilon_s:
                    s_action = np.random.randint(0, self.env.NR_OF_ROBOT_ACTIONS)
                else:
                    # Handles the case where Q-values are equal (as at the beginning)
                    max_val = np.max(current_student_QTable[current_index, :])
                    max_actions = [a for a in range(self.env.NR_OF_ROBOT_ACTIONS) if current_student_QTable[current_index, a] == max_val]
                    s_action = np.random.choice(max_actions)
                    
                # Check if the student (robot) is on "unpreferred cells" (not preferred by human)
                color = self.env.check_if_agent_is_on_unpreferred_cell(self.teacher.MODEL_OF_HUMAN_COLORS)    

                # Take the action and observe the outcome
                new_obs, reward_s, ep_terminated, ep_truncated, info = self.env.step(s_action,t_action,color,self.teacher.MODEL_OF_HUMAN_COLORS)
                
                next_state = (*self.env.agent_pos, self.env.agent_dir)
                next_index = self.state_to_index(next_state, self.env.width, self.env.NR_OF_ROBOT_DIRECTIONS)
     
                # Update the Q-value
                current_student_QTable[current_index, s_action] += self.cfg[ConfigManager.ALPHA_S] * (reward_s + self.cfg[ConfigManager.GAMMA_S] * np.max(current_student_QTable[next_index, :]) 
                                                                           - current_student_QTable[current_index, s_action])
                cumulative_reward_s += reward_s
                
                # Move to the next state
                current_state = next_state
                current_index = next_index    
                       
            # ======================= END STUDENT LEARNING PHASE ==================================
                
            # ======================= UPDATE TEACHER ACTION VALUES  ===============================
                                
            reward_teacher = self.teacher._reward_Human(t_action, info.get('r_tau'), self.env.cell_visit_frequencies) # Compute the human reward based on selected action
            self.teacher_Q_Values[t_action] = (1-self.cfg[ConfigManager.ALPHA_T]) * self.teacher_Q_Values[t_action] + self.cfg[ConfigManager.ALPHA_T] * reward_teacher # Update the human Q-values

            if ep_terminated == True: # the agent reached the goal (terminal state)
                self.student_competence.append(1)
            else:
                self.student_competence.append(0)
            
            if t_action == self.teacher.HUMAN_ACTION_STAY:
                self.cumulative_teacher_actions.append(1)
            else:
                self.cumulative_teacher_actions.append(0)
            
            self.cumulative_reward_s_trend.append(cumulative_reward_s)
            self.cumulative_reward_teacher.append(reward_teacher)
             
        # ======================= END EPISODE ==================================
        
        # Close the environment
        self.env.close()