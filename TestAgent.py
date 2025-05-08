import gymnasium as gym
from BridgeEnv import BridgeEnv
from BridgeAgentQL import QLearningAgent


def run_episode(env, agent, time_label, time_flag):
    # Reset environment and set time of day manually
    obs, _ = env.reset()
    env.time_of_day = time_flag

    # Set object position based on time of day (morning/evening)
    if time_flag == 0:
        env.object_pos = env.object_loc_morning
    else:
        env.object_pos = env.object_loc_evening

    # Reinitialize grid with updated object and time
    env.grid = env._init_grid()
    obs = env._get_obs()

    state = agent._discretize(obs)
    done = False
    step_count = 0

    print(f"Episode start: {time_label}")
    while not done and step_count < env.max_steps:
        # Use greedy policy to select action from Q-table
        action = agent.choose_action_test(state, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()  # display each step visually

        done = terminated or truncated
        state = agent._discretize(obs)
        step_count += 1

    print(f"Episode end: {time_label} | Steps: {step_count} | Reward: {reward}\n")


def main():
    # Build test environment with human rendering
    eval_env = BridgeEnv(grid_size=8, render_mode="human", phase="train")

    # Create agent and load pretrained Q-table
    agent = QLearningAgent(eval_env)
    agent.load("q_table.pkl")

    # Run two test rollouts: one in morning, one in evening
    run_episode(eval_env, agent, time_label="Morning", time_flag=0)
    run_episode(eval_env, agent, time_label="Evening", time_flag=1)

    eval_env.close()


if __name__ == "__main__":
    main()
