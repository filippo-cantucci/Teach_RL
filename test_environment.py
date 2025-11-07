from minigrid.manual_control import ManualControl
from environment import MyEnvironment
from configManager import ConfigManager, ConfigConstants


if __name__ == "__main__":
    
    params_to_override = {ConfigConstants.RENDER_MODE.value: "human"}
    params = ConfigManager.load_config("config.yaml", overrides=params_to_override)
        
    env = MyEnvironment(params)
    manual_control = ManualControl(env, seed=params[ConfigConstants.SEED.value])
    manual_control.start()
    
