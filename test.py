import numpy as np
import os
import sys
from environment import MyEnvironment
from fixed_environment import FixedEnvironment
from logger import setup_logging, get_logger
from configManager import ConfigConstants
from configManager import ConfigManager

# Setup logging
setup_logging()
logger = get_logger("test")


def state_to_index(state, size, dir_max):
    # Convert the state (y, x, d) to a unique index
    y, x, d = state
    return (y * size + x) * dir_max + d


params_to_override = {ConfigConstants.RENDER_MODE: "human"}
params = ConfigManager.load_config("config.yaml", params_to_override)

# Construct the path to the Results directory
results_dir = f"results/Results_0.0/Q_Tables/"
q_table_path = os.path.join(results_dir, "Q_table_0.csv")

logger.info(f"Loading Q-table from: {q_table_path}")
print(f"Loading Q-table from: {q_table_path}")

# Check if the file exists
if not os.path.exists(q_table_path):
    error_msg = f"Error: The Q-table file was not found at {q_table_path}"
    logger.error(error_msg)
    print(error_msg)
    
    print("Available result directories:")
    available_dirs = []
    for item in os.listdir("results"):
        if item.startswith("Results_"):
            available_dirs.append(item)
            print(f"  - {item}")
    
    logger.error(f"Available result directories: {available_dirs}")
    sys.exit(1)

# Load the Q-table from the Results directory
q_table = np.loadtxt(q_table_path, delimiter=",")
logger.info(f"Q-table loaded successfully. Shape: {q_table.shape}")

# Initialize the environment with rendering mode set to "human"

env = FixedEnvironment(params)
logger.info("Environment initialized for testing with human rendering")

# Reset the environment and get the initial observation
obs, _ = env.reset()
logger.debug(f"Environment reset. Initial agent position: {env.agent_pos}, direction: {env.agent_dir}")

terminated = False
truncated = False
dir_nr = 4  # Number of possible directions

# Run the loop until the episode is terminated or truncated
step_count = 0
logger.info("Starting test episode with trained agent")

while not terminated and not truncated:
    step_count += 1
    # Get the current state of the agent
    current_state = *env.agent_pos, env.agent_dir
    
    # Convert the current state to an index for the Q-table
    current_index = state_to_index(current_state, env.width, dir_nr)
    q_values = q_table[current_index]
    action = np.argmax(q_values)
    
    logger.debug(f"Step {step_count}: Agent at position {env.agent_pos}, direction {env.agent_dir}")
    logger.debug(f"Q-values for current state: {q_values}, selected action: {action}")
    
    # Take the action and get the new observation and reward
    obs, reward, terminated, truncated, info = super(type(env), env).step(action)
    
    # Log if agent is on a red cell
    if env._agent_is_on_color_cell('red'):
        logger.info(f"Step {step_count}: Agent is on a red cell at position {env.agent_pos}")
    
    # Render the environment
    env.render()

# Episode completed
if terminated:
    logger.info(f"Test episode completed: Agent reached the goal in {step_count} steps")
elif truncated:
    logger.info(f"Test episode truncated after {step_count} steps (max steps reached)")

# Close the environment
env.close()
logger.info("Test completed")