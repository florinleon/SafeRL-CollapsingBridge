import gymnasium as gym
import numpy as np
import pickle
import os
from BridgeEnv import BridgeEnv


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # decay rate for exploration
        self.min_epsilon = min_epsilon  # floor value for epsilon

        self.use_masking = True  # flag for hard action masking

        self.q_table = {}  # maps state -> action values
        self.allowed_actions = {}  # maps state -> allowed actions (for masking)


    def _discretize(self, obs):
        # Convert observation into a hashable state representation
        grid_flat = tuple(obs["local_grid"].flatten())
        pos = tuple(obs["agent_pos"])
        time = obs["time_indicator"]
        carrying = obs["has_object"]
        return (grid_flat, pos, time, carrying)


    def choose_action(self, state, obs):
        # ε-greedy action selection with optional hard masking
        if state not in self.allowed_actions:
            self.allowed_actions[state] = list(range(self.env.action_space.n))
        actions = self.allowed_actions[state]
        if not actions:
            raise Exception(f"No allowed actions left for state {state}")
        q_vals = self.q_table.get(state, np.zeros(self.env.action_space.n))
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        sorted_actions = np.argsort(q_vals)[::-1]
        for act in sorted_actions:
            if act in actions:
                return int(act)
        return np.random.choice(actions)


    def choose_action_test(self, state, obs):
        # Greedy action selection (used during evaluation)
        q_vals = self.q_table.get(state, np.zeros(self.env.action_space.n))
        return int(np.argmax(q_vals))


    def update(self, state, action, reward, next_state, done, death_cause=None):
        # Standard Q-learning update
        q_vals = self.q_table.setdefault(state, np.zeros(self.env.action_space.n))
        next_q_vals = self.q_table.get(next_state, np.zeros(self.env.action_space.n))
        td_target = reward + self.gamma * np.max(next_q_vals) * (not done)
        q_vals[action] += self.alpha * (td_target - q_vals[action])

        # Remove unsafe actions if agent dies from stepping into water
        if self.use_masking and done and death_cause == "bad_end":
            if action in self.allowed_actions.get(state, []):
                self.allowed_actions[state].remove(action)


    def test_agent(self):
        # Runs test episodes in both time conditions
        success = True
        for t in [0, 1]:  # 0=morning, 1=evening
            test_env = BridgeEnv(grid_size=self.env.unwrapped.grid_size, render_mode=None, phase="train")
            obs, _ = test_env.reset()
            test_env.time_of_day = t
            test_env.object_pos = test_env.object_loc_morning if t == 0 else test_env.object_loc_evening
            test_env.grid = test_env._init_grid()
            obs = test_env._get_obs()
            state = self._discretize(obs)
            steps = 0
            done = False
            while not done and steps < test_env.max_steps:
                action = self.choose_action_test(state, obs)
                obs, reward, term, trunc, _ = test_env.step(action)
                done = term or trunc
                state = self._discretize(obs)
                steps += 1
                if done and reward > 0:
                    print(steps)
                    break
            else:
                print(f"Test time={'morning' if t==0 else 'evening'}: failed")
                success = False
        return success


    def train(self, episodes, reward_log_path="q_learning_rewards.txt"):
        # Main training loop
        with open(reward_log_path, "w") as log_file:
            for ep in range(episodes):
                obs, _ = self.env.reset()
                state = self._discretize(obs)
                total_reward = 0
                done = False

                while not done:
                    action = self.choose_action(state, obs)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state = self._discretize(next_obs)

                    cause = ""
                    if done:
                        if reward > 0:
                            cause = "success"
                        elif truncated or self.env.unwrapped.steps >= self.env.unwrapped.max_steps:
                            cause = "max_steps"
                        else:
                            cause = "bad_end"
                            self.log_bad_end(state, action)

                    self.update(state, action, reward, next_state, done, death_cause=cause)
                    state = next_state
                    obs = next_obs
                    total_reward += reward

                log_file.write(f"{total_reward}\n")
                log_file.flush()
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                print(f"Episode {ep+1}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

                # Periodic convergence check
                if (ep+1) % 500 == 0:
                    if self.test_agent():
                        print(f"Converged at episode {ep+1}. Stopping early.")
                        with open("converged.txt", "a") as f:
                            f.write(f"Converged at episode {ep+1}. Stopping early.\n")
                        break

        self.dump_q_values("q_values.txt")
        self.count_unsafe_deaths()


    def count_unsafe_deaths(self):
        # Count episodes with unsafe terminal states
        with open("q_learning_rewards.txt", "r") as f:
            returns = [float(line.strip()) for line in f if line.strip()]
        deaths = [1 if -1099 <= r <= -1000 else 0 for r in returns]
        deaths_array = np.array(deaths)
        total_deaths = np.sum(deaths_array)
        total_episodes = len(returns)
        print(f"Unsafe deaths: {total_deaths} / {total_episodes} ({100.0 * total_deaths / total_episodes:.2f}%)")


    def log_bad_end(self, state, action):
        # Log details about bad end (fall into water) for later review
        action_names = {0: "left", 1: "right", 2: "up", 3: "down"}
        cell_names   = { 0: "wall", 1: "floor", 2: "water", 3: "bridge", 4: "time", 5: "object", 6: "goal" }

        grid_flat, pos, time_flag, carrying = state
        named = [cell_names.get(c, str(c)) for c in grid_flat]
        grid_part = ",".join(named)

        time_str   = "morning" if time_flag == 0 else "evening"
        carry_str  = "carrying" if carrying else "empty"
        action_str = action_names.get(action, str(action))

        line = (f"{grid_part}; pos=({pos[0]},{pos[1]}); time={time_str}; carry={carry_str}; action={action_str}\n")
        with open("unsafe_actions_log.txt", "a") as f:
            f.write(line)


    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)


    def load(self, path="q_table.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)


    def dump_q_values(self, filename):
        # Write Q-values and allowed actions for inspection/debugging
        action_names = {0: "left", 1: "right", 2: "up", 3: "down"}
        with open(filename, "w") as f:
            for state, q_vals in self.q_table.items():
                _, pos, time_flag, carrying = state
                pos = tuple(int(x) for x in pos)
                time_str = "morning" if time_flag == 0 else "evening"
                carry_str = "carrying" if carrying else "empty"
                f.write(f"State  pos={pos}  time={time_str}  {carry_str}\n")
                allowed = self.allowed_actions.get(state, list(range(self.env.action_space.n)))
                for a in allowed:
                    name = action_names.get(a, str(a))
                    f.write(f"    {name:>5}:  {q_vals[a]:.4f}\n")
                f.write("\n")


if __name__ == "__main__":
    # Train and serialize a Q-learning agent in the bridge environment
    env = gym.make("BridgeEnv-v0", render_mode=None, phase="train")
    agent = QLearningAgent(env)
    agent.train(episodes=25000)
    agent.save()
