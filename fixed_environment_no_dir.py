from __future__ import annotations
from minigrid.core.grid import Grid
from my_actions import MyActions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Floor
from minigrid.minigrid_env import MiniGridEnv
from logger import get_logger
from human import HumanConstants
from configManager import ConfigConstants

class MyEnvConstants:
    NR_OF_AGENT_DIRECTIONS = 0 # len(DIR_TO_VEC) # 0: right, 1: down, 2: left, 3: up
    NR_OF_AGENT_ACTIONS =  len([MyActions.go_down, MyActions.go_up, MyActions.go_right, MyActions.go_left]) #  4 Actions

class FixedEnvironment(MiniGridEnv):
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
                
        # print(f"ALPHA REW: {self.alpha}")    
                
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
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Init the starting random position for the agent (avoid edges)
        self.agent_start_pos = (1,1)

        # Generate vertical separation wall with a gap
        wall_fixed_offset = (height -2) - round((width) / 3)
        gap_pos_down_1 = 1
        gap_pos_down_2 = 2 # width - round(width / 3)
        for i in range(0, height):
            if i != gap_pos_down_1 and i != gap_pos_down_2:
                self.grid.set(wall_fixed_offset, i, Wall())
            else:
                self.grid.set(wall_fixed_offset, i, None)
                
        # Place red cells (preferences) in front of the gap the grid
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),6,1)         
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),6,2)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),6,3)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),6,4)  
        
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),5,1)         
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),5,2)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),5,3)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),5,4)        

        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),10,5)         
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),10,6)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),10,7)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),10,8)            

        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),11,5)         
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),11,6)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),11,7)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),11,8)   
        
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),12,5)         
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),12,6)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),12,7)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),12,8)            

        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),13,5)         
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),13,6)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),13,7)
        self.put_obj(Floor(HumanConstants.HUMAN_COLOR_PREF),13,8)           
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height-2)

        # Place the agent in the starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
        #else:
        #    self.place_agent()
            

    def step(self, action, teacher_action=None, r_p=0.0):
        
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Calculate new position based on action
        current_pos = self.agent_pos
        new_pos = None

        # Left
        if action == MyActions.go_left:
            new_pos = (current_pos[0] - 1, current_pos[1])
        
        # Right
        elif action == MyActions.go_right:
            new_pos = (current_pos[0] + 1, current_pos[1])
        
        # Up
        elif action == MyActions.go_up:
            new_pos = (current_pos[0], current_pos[1] - 1)
        
        # Down
        elif action == MyActions.go_down:
            new_pos = (current_pos[0], current_pos[1] + 1)
	
        # Done action (not used by default)
        elif action == MyActions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        # Check if new position is valid and move agent
        if new_pos is not None:
            # Check if new position is within grid bounds
            if (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
                # Get the contents of the target cell
                target_cell = self.grid.get(*new_pos)
                
                # Check if the cell can be moved to (not a wall)
                if target_cell is None or target_cell.can_overlap():
                    self.agent_pos = new_pos
                    
                    # Check if agent reached the goal
                    if target_cell is not None and target_cell.type == "goal":
                        terminated = True
                        reward = self._reward()

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        
        r_ag = 0.0
                                     
        if teacher_action == HumanConstants.ACTION_STAY:
            if self._agent_is_on_color_cell(HumanConstants.HUMAN_COLOR_PREF):
                r_ag = r_tau + r_p
                self.r_mp += self.alpha * (r_p - self.r_mp)  # update the model with the preference reward
                # print(f"STUDENT -- IN STAY AND RED CELL: R_AG {r_ag} - R_TAU {r_tau} - R_MP {self.r_mp}")
            else:
                r_ag = r_tau
                # print(f"STUDENT -- IN STAY AND EMPTY CELL: R_AG {r_ag} - R_TAU {r_tau} - R_MP {self.r_mp}")
        else:
            if self._agent_is_on_color_cell(HumanConstants.HUMAN_COLOR_PREF):
                r_ag = r_tau + self.r_mp
                # print(f"STUDENT -- IN LEAVE AND RED CELL: R_AG {r_ag} - R_TAU {r_tau} - R_MP {self.r_mp}")
            else:
                r_ag = r_tau
                # print(f"STUDENT -- IN LEAVE AND EMPTY CELL: R_AG {r_ag} - R_TAU {r_tau} - R_MP {self.r_mp}")
                            
        self.logger.debug("Student rewards: r_tau=%.3f, r_ag=%.3f", r_tau, r_ag)

        info['r_tau'] = r_tau
        info['r_mp'] = self.r_mp
                        
        return obs, r_ag, terminated, truncated, info
