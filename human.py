from configManager import ConfigManager
from types import MappingProxyType
from minigrid.core.constants import COLOR_NAMES

class Human():
     
    MODEL_OF_HUMAN_COLORS = MappingProxyType({ 
                               "blue"   : -0.2,   
                               "green"  : -0.3,      
                               "grey"   : -0.15,  
                               "purple" : -0.18,  
                               "red"    : -0.1,   
                               "yellow" : -0.2})  
    
    HUMAN_ACTION_STAY = 'stay'
    HUMAN_ACTION_LEAVE = 'leave'

    def __init__(
        self,
        params,
    ):
        print("Human INIT ")
        self.cfg = params
        
    def _reward_Human(self, human_action, r_tau, cell_visited = None):
                
        # Inizializzazione componenti
        reward_human = 0.0  # total human reward
        reward_to_leave = 1.0  # bonus quando l'umano decide di "leave"
        reward_preferences = 0.0  # componente legata alle preferenze sui colori

        # Se non viene fornita la mappa delle celle visitate, la preferenza è nulla
        if not cell_visited:
            reward_preferences = 0.0
        else:
            # Somma pesata: per ogni colore configurato, moltiplica il peso
            # di preferenza per il numero di celle visitate di quel colore
            reward_preferences = sum(
                self.MODEL_OF_HUMAN_COLORS[h_p] * cell_visited[h_p]
                for h_p in self.MODEL_OF_HUMAN_COLORS
                if h_p in cell_visited
            )
                    
        # Combinazione finale in base all'azione umana
        if human_action == self.HUMAN_ACTION_STAY:
            reward_human = r_tau + reward_preferences
        else:
            reward_human = r_tau + self.cfg[ConfigManager.ABSENCE_MUX] * reward_preferences + reward_to_leave
            
        return reward_human
   
"""
def main():
        params = ConfigManager.load_config("config.yaml")
        cell_visited = {'red': 1, 'blue': 2}
        h = Human(params)
        p = h._reward_Human('leave',1,cell_visited)
        print(p)
        
if __name__ == "__main__":
        main()
"""