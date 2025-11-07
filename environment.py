from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Floor
from minigrid.minigrid_env import MiniGridEnv
from configManager import ConfigManager
from minigrid.core.constants import COLOR_NAMES
from human import Human


class MyEnvironment(MiniGridEnv):
    
    NR_OF_ROBOT_DIRECTIONS = 4 # 0: right, 1: down, 2: left, 3: up
    NR_OF_ROBOT_ACTIONS =  3 # Number of agent (student) actions (forward, left, right)
    
    BLUE, GREEN, GREY, PURPLE, RED, YELLOW = COLOR_NAMES

    def __init__(
        self,
        params,
        **kwargs,
    ):
        print("Environment INIT ")
        
        # attributi di environment
        self.cfg = params
        self.size = params[ConfigManager.GRID_SIZE]
        self.layouts = {
            "v1": self.build_v1,
            "v2": self.build_v2,
            "v3": self.build_v3,
            "v4": self.build_v4,
            }
        
        # attributi di agent
        self.estimated_model_of_human_colors = dict.fromkeys(Human.MODEL_OF_HUMAN_COLORS, 0.0)   
        
        # Define the mission space
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Initialize the MiniGrid environment
        super().__init__(
            mission_space=mission_space,
            grid_size=self.cfg[ConfigManager.GRID_SIZE],
            see_through_walls=True, # Set this to True for maximum speed
            max_steps=self.cfg[ConfigManager.MAX_STEPS],
            render_mode =self.cfg[ConfigManager.RENDER_MODE],
            **kwargs,
        )
        
    def _is_on_unpreferred_cell(self, human_preferences=None):
        
        if human_preferences:
           color = getattr(self.grid.get(*self.agent_pos), "color", None) if self.grid.get(*self.agent_pos) is not None else None
           return color if (color in human_preferences) else None
        else:
            return None
    
    def _update_model_of_h_pref(self,color,human_preferred_colors):
        
        self.estimated_model_of_human_colors[color] += self.cfg[ConfigManager.ALPHA_REW_MODEL] * (human_preferred_colors[color]
                                                                - self.estimated_model_of_human_colors[color])

    def rebuild_env(self, layout,width, height):
        print("Rebuild Environment ",layout)
        self._gen_grid(width, height)
    
    @staticmethod
    def _gen_mission():
        # Generate a mission description
        return "Reach the green Goal"
    
    def _gen_grid(self,width, height):
                
        make = self.layouts.get(self.cfg[ConfigManager.LAYOUT_V], self.build_v1)
        make(width,height)
    
    # Layout v1
    """
    Caratteristiche:
        - agente parte da posizione (1,1)
        - goal in basso a destra
        - muro divisore a metà con gap all'inizio
        -solo 3 blocchi di celle ROSSE, uno vicino alla posizione di partenza e due verso il goal
    """
    def build_v1(self, width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Init the starting random position for the agent (avoid edges)
        self.agent_start_pos = (1,1)
        self.agent_start_dir = 0  # 0: right, 1: down, 2: left, 3: up

        # Generate vertical separation wall with a gap
        wall_fixed_offset = (self.size -2) - round((self.size) / 3)
        gap_pos_down_1 = 1
        gap_pos_down_2 = 2
        for i in range(0, self.size):
            if i != gap_pos_down_1 and i != gap_pos_down_2:
                self.grid.set(wall_fixed_offset, i, Wall())
            else:
                self.grid.set(wall_fixed_offset, i, None)
                
        # Place red cells (preferences) in front of the gap the grid
        for i in range(1,9):
            self.put_obj(Floor(self.RED),wall_fixed_offset // 2,i)
            self.put_obj(Floor(self.RED),(wall_fixed_offset // 2)+1,i)
        for i in range(1,6):
            self.put_obj(Floor(self.RED),wall_fixed_offset+i,width // 3 + 6)
            self.put_obj(Floor(self.RED),wall_fixed_offset+i,width // 3 + 5) 
            if i >= 2:
                self.put_obj(Floor(self.RED),width-i,width // 3) 
                self.put_obj(Floor(self.RED),width-i,width // 3 + 1)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height-2)

        # Place the agent in the starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    """
    Caratteristiche:
        - agente parte da posizione (1,1)
        - goal in basso a destra
        - muro divisore a metà con gap all'inizio
        -solo 3 blocchi di celle BLU, uno vicino alla posizione di partenza e due verso il goal
    """
    def build_v2(self,width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Init the starting random position for the agent (avoid edges)
        self.agent_start_pos = (1,1)
        self.agent_start_dir = 0  # 0: right, 1: down, 2: left, 3: up

        # Generate vertical separation wall with a gap
        wall_fixed_offset = (self.size -2) - round((self.size) / 3)
        gap_pos_down_1 = 1
        gap_pos_down_2 = 2
        gap_pos_down_3 = height - 2
        gap_pos_down_4 = height - 3
        for i in range(0, self.size):
            if i != gap_pos_down_1 and i != gap_pos_down_2 and i != gap_pos_down_3 and i != gap_pos_down_4:
                self.grid.set(wall_fixed_offset, i, Wall())
            else:
                self.grid.set(wall_fixed_offset, i, None)
                
        # Place red cells (preferences) in front of the gap the grid
        for i in range(1,8):
            self.put_obj(Floor(self.BLUE),wall_fixed_offset // 2,i)
            self.put_obj(Floor(self.BLUE),(wall_fixed_offset // 2)+1,i)
            
            self.put_obj(Floor(self.RED),wall_fixed_offset // 2,height - i - 1)
            self.put_obj(Floor(self.RED),(wall_fixed_offset // 2)+1,height - i - 1)
        for i in range(1,6):
            self.put_obj(Floor(self.BLUE),wall_fixed_offset+i,width // 3 + 6)
            self.put_obj(Floor(self.BLUE),wall_fixed_offset+i,width // 3 + 5) 
            self.put_obj(Floor(self.RED),width-i - 1,width // 3) 
            self.put_obj(Floor(self.RED),width-i - 1,width // 3 + 1)
        
        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height-2)
        self.put_obj(Goal(), width -2, height // 2 - 1)


        # Place the agent in the starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()    
     
    """
    Caratteristiche:
        - agente parte da posizione (1,width/2)
        - goal in posizione (height, width/2)
        - muro divisore a metà con gap al centro
        -due blocchi di celle una rossa a sx del muro, una a dx del muro, entrambi nella parte centrale
    """                       
    def build_v3(self,width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Init the starting random position for the agent (avoid edges)
        self.agent_start_pos = (1,width // 2)
        self.agent_start_dir = 0  # 0: right, 1: down, 2: left, 3: up

        # Generate vertical separation wall with a gap
        wall_fixed_offset = (self.size -3) - round((self.size) / 3)
        gap_pos_down_1 = width // 2
        gap_pos_down_2 = gap_pos_down_1 + 1
        for i in range(0, self.size):
            if i != gap_pos_down_1 and i != gap_pos_down_2:
                self.grid.set(wall_fixed_offset, i, Wall())
            else:
                self.grid.set(wall_fixed_offset, i, None)
                
        # Place red cells (preferences) in front of the gap the grid
        for i in range(2,9):
            self.put_obj(Floor(self.BLUE),wall_fixed_offset // 2,i + 5)
            self.put_obj(Floor(self.BLUE),(wall_fixed_offset // 2)+1,i + 5)
            self.put_obj(Floor(self.BLUE),(wall_fixed_offset // 2)+2,i + 5)
        for i in range(2,9):
            self.put_obj(Floor(self.RED),(wall_fixed_offset // 2) + 8,i + 5)
            self.put_obj(Floor(self.RED),(wall_fixed_offset // 2)+9,i + 5)
            self.put_obj(Floor(self.RED),(wall_fixed_offset // 2)+10,i + 5)
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width -2, height // 2)

        # Place the agent in the starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()    
    
        """
    Caratteristiche:
        - agente parte da posizione in alto al centro
        - due blocchi di muri agli angoli sx e dx
        - goal in posizione in basso al centro
        -un blocco centrale di celle rosse e blu
    """       
    def build_v4(self,width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Init the starting random position for the agent (avoid edges)
        self.agent_start_pos = ((width // 2 - 1), 1)
        self.agent_start_dir = 1  # 0: right, 1: down, 2: left, 3: up
        
        for i in range(1,7):
            for j in range(1,7):
                self.put_obj(Floor(self.BLUE), i, j)
                self.put_obj(Floor(self.RED), height - 1 - i, j)

                
        # Place red cells (preferences) in front of the gap the grid
        for i in range(1,12):
            self.put_obj(Floor(self.BLUE), i + 3,8)
            self.put_obj(Floor(self.RED), i + 3,9)
            self.put_obj(Floor(self.BLUE), i + 3,10)
            self.put_obj(Floor(self.RED), i + 3,11)
            self.put_obj(Floor(self.BLUE), i + 3,12)
            self.put_obj(Floor(self.RED), i + 3,13)
            
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), (width // 2) - 1, height - 2)

        # Place the agent in the starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()         
            
            
    def step(self, action, human_action=None, color=None, human_color_preferences=None):
                
        obs, r_tau, terminated, truncated, info = super().step(action)   
        
        r_ag = 0.0
                            
        if human_action == Human.HUMAN_ACTION_STAY:
            if color:
                r_ag = r_tau + human_color_preferences[color]
                self._update_model_of_h_pref(color, human_color_preferences)  # update the model with the preference reward
            else:
                r_ag = r_tau
        else:
            if color:
                r_ag = r_tau + self.estimated_model_of_human_colors[color]    
            else:
                r_ag = r_tau
                            
        info['r_tau'] = r_tau
                                
        return obs, r_ag, terminated, truncated, info
    
    
    
"""    
from minigrid.manual_control import ManualControl
from environment import MyEnvironment
    
if __name__ == "__main__":
    
    params_to_override = {ConfigManager.RENDER_MODE: "human"}
    params = ConfigManager.load_config("config.yaml", overrides=params_to_override)
        
    env = MyEnvironment(params)
    manual_control = ManualControl(env, seed=params[ConfigManager.SEED])
    manual_control.start()
"""   