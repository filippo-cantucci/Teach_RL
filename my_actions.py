from minigrid.core.actions import Actions

from enum import IntEnum


class MyActions():
    # Go Up, Go Down, Go Left, Go Right (No pose, only position) 
    
    # Go to the left position
    go_left = 0
    
    # Go to the right position
    go_right = 1
    
    # Go to the up position
    go_up = 2
    
    # Go to the down position
    go_down = 3
    
    # Done completing task 
    done = 4