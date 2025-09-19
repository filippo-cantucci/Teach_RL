from __future__ import annotations
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Floor
from minigrid.minigrid_env import MiniGridEnv
from logger import get_logger
from human import HumanConstants
from configManager import ConfigConstants

class MyEnvConstants:
    NR_OF_AGENT_DIRECTIONS = len(DIR_TO_VEC) # 0: right, 1: down, 2: left, 3: up
    NR_OF_AGENT_ACTIONS =  len([Actions.forward,Actions.left,Actions.right]) #  3 # Number of agent (student) actions (forward, left, right)

class MyEnvironment(MiniGridEnv):
    def __init__(
        self,
        params,
        **kwargs,
    ):
        # Initialize agent's starting position and direction
        
        self.alpha = params[ConfigConstants.ALPHA_REW_MODEL]
        self.size = params[ConfigConstants.GRID_SIZE]
        self.max_steps = params[ConfigConstants.MAX_STEPS]  # max nr of steps for each episode
        self.render_m = params[ConfigConstants.RENDER_MODE]  # grid render visualization mode
                        
        self.r_mp = 0.0  # reward model based on teacher preferences built by the agent
                
        # Initialize logger
        self.logger = get_logger("environment")
        
        # Define the mission space
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Set the maximum number of steps if not provided
        if self.max_steps is None:
            self.max_steps = 4 * self.size**2

        # Initialize the MiniGrid environment
        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            see_through_walls=True, # Set this to True for maximum speed
            max_steps=self.max_steps,
            render_mode=self.render_m,
            **kwargs,
        )
                
    def _agent_is_on_color_cell(self,color):
        isOn = False
        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.color == color:
            isOn = True
        return isOn

    @staticmethod
    def _gen_mission():
        # Generate a mission description
        return "Reach the green Goal"
    
    def _gen_grid(self, width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)
                
        # Init the starting random position for the agent (avoid edges)
        self.agent_start_pos = (1, self._rand_int(1,height-1))
        self.agent_start_dir = 0  # 0: right, 1: down, 2: left, 3: up
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
                
        # Place the goal as far as possible from the agent starting position
        self.put_obj(Goal(), width - 2, self._rand_int(1,height-1))
        
        # Place the agent in the starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        # Place red cells (preferences) in empty zones of the grid
        for _ in range(self._rand_int(1,self.size)):
            self.place_obj(Floor(HumanConstants.HUMAN_COLOR_PREF))     
        
        # Place walls cell (obstacles) in empy zones of the grid
        for _ in range(self._rand_int(1,self.size)):
            self.place_obj(Wall())
            
        # Remove eventual obstacle in front of it (if exists)
        self.grid.set(self.agent_start_pos[0]+1, self.agent_start_pos[1], None)
        

    def step(self, action, teacher_action=None, r_p=0.0):
        
        self.logger.debug("Step of Agent: student_action=%s, teacher_action=%s", action, teacher_action)
        obs, r_tau, terminated, truncated, info = super().step(action)   
        
        r_ag = 0.0
                                     
        if teacher_action == HumanConstants.ACTION_STAY:
            if self._agent_is_on_color_cell(HumanConstants.HUMAN_COLOR_PREF):
                r_ag = r_tau + r_p
                self.r_mp += self.alpha * (r_p - self.r_mp)  # update the model with the preference reward
            else:
                r_ag = r_tau
        else:
            if self._agent_is_on_color_cell(HumanConstants.HUMAN_COLOR_PREF):
                r_ag = r_tau + self.r_mp
            else:
                r_ag = r_tau
                            
        self.logger.debug("Student rewards: r_tau=%.3f, r_ag=%.3f", r_tau, r_ag)

        info['r_tau'] = r_tau
        info['r_mp'] = self.r_mp
                        
        return obs, r_ag, terminated, truncated, info
