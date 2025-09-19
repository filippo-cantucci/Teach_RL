from minigrid.manual_control import ManualControl
from environment import MyEnvironment
from configManager import ConfigManager, ConfigConstants
from fixed_environment import FixedEnvironment


if __name__ == "__main__":
    
    params_to_override = {ConfigConstants.RENDER_MODE: "human"}
    params = ConfigManager.load_config("config.yaml", overrides=params_to_override)
        
    env = FixedEnvironment(params)
    manual_control = ManualControl(env, seed=params[ConfigConstants.SEED])
    manual_control.start()
    
