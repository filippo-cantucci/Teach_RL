from logger import get_logger
from minigrid.core.constants import COLORS, COLOR_NAMES
from configManager import ConfigConstants

class HumanConstants:
    
    ABSENCE_MUX = 1
    ACTION_STAY = 'stay'
    ACTION_LEAVE = 'leave'
    HUMAN_COLOR_PREF = 'red'

class Human():
    def __init__(
        self,
        params,
    ):
        # Initialize logger
        self.logger = get_logger("human")
 
        self.absence_multiplier =  HumanConstants.ABSENCE_MUX
        self.r_preference = params[ConfigConstants.PREFERENCE_VAL_RED]   

    def _rewardT(self, h_action, r_tau, visited):
        
        r_teach = 0.0
        r_pref = visited * self.r_preference
        r_to_leave = 1.0
        
        if h_action == HumanConstants.ACTION_STAY:
            r_teach = r_tau + r_pref
        else:
            r_teach = r_tau + (self.absence_multiplier * r_pref) + r_to_leave
                        
        self.logger.debug("Calculating teacher reward: action %s, r_teach = r_tau + r_pref = %.3f + %.3f = %.3f", 
                             h_action, r_tau, r_pref, r_teach)
            
        return r_teach
    

