import gymnasium as gym
import numpy as np
import pickle
import os
from BridgeEnv import BridgeEnv


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.env = env
        self.alpha = alpha              # learning rate
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # initial exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon  # minimum allowed epsilon

        self.q_table = {}               # Q-values: maps state → action values
        self.tau = 1000                 # safety schedule scale for risk-awareness

        # safety statistics: per-(state, action)
        self.visit_counts = {}          # number of times each (s,a) was tried
        self.death_counts = {}          # number of times each (s,a) led to unsafe termination

        self.current_episode = 0        # for w(e) schedule


    def _discretize(self, obs):
        # Convert observation into a tuple key used in Q-table and count tracking
        grid_flat = tuple(obs["local_grid"].flatten())
        pos = tuple(obs["agent_pos"])
        time = obs["time_indicator"]
        carrying = obs["has_object"]
        return (grid_flat, pos, time, carrying)


    def choose_action(self, state, obs):
        # Initialize safety count structures if needed
        if state not in self.visit_counts:
            n = self.env.action_space.n
            self.visit_counts[state] = np.zeros(n, dtype=int)
            self.death_counts[state] = np.zeros(n, dtype=int)

        # Get Q-values for the current state
        q_vals = self.q_table.get(state, np.zeros(self.env.action_space.n))

        # Standard epsilon-greedy base distribution
        n = self.env.action_space.n
        pi0 = np.ones(n) * (self.epsilon / n)
        best = int(np.argmax(q_vals))
        pi0[best] += (1.0 - self.epsilon)

        # Compute risk-awareness weight: w(e) increases over episodes
        e = self.current_episode
        w = 1.0 - np.exp(-e / self.tau)  # smoothly grows from 0 to 1

        # Estimate empirical death probabilities
        visits = self.visit_counts[state]
        deaths = self.death_counts[state]
        p_death = np.divide(deaths, np.maximum(visits, 1))  # avoid div-by-zero

        # Modify policy to down-weight high-risk actions
        modifier = 1.0 - w * p_death
        adjusted = pi0 * modifier
        total = adjusted.sum()
        if total <= 0:
            adjusted = pi0  # fallback to unmodified if all probabilities collapse
            total = adjusted.sum()

        probs = adjusted / total  # re-normalize
        return int(np.random.choice(np.arange(n), p=probs))  # sample action


    def choose_action_test(self, state, obs):
        # Greedy policy for evaluation
        q_vals = self.q_table.get(state, np.zeros(self.env.action_space.n))
        return int(np.argmax(q_vals))


    def update(self, state, action, reward, next_state, done, death_cause=None):
        # Update safety statistics
        self.visit_counts[state][action] += 1
        if death_cause == "bad_end":
            self.death_counts[state][action] += 1

        # Standard Q-learning update
        q_vals = self.q_table.setdefault(state, np.zeros(self.env.action_space.n))
        next_q = self.q_table.get(next_state, np.zeros(self.env.action_space.n))
        target = reward + self.gamma * np.max(next_q) * (not done)
        q_vals[action] += self.alpha * (target - q_vals[action])


    def test_agent(self):
        # Run one evaluation in morning and evening
        success = True
        for t in [0, 1]:
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
                self.current_episode = ep  # update schedule counter
                obs, _ = self.env.reset()
                state = self._discretize(obs)
                total_reward = 0
                done = False

                while not done:
                    action = self.choose_action(state, obs)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state = self._discretize(next_obs)

                    # Diagnose termination type
                    cause = ""
                    if done:
                        if reward > 0:
                            cause = "success"
                        elif truncated or self.env.unwrapped.steps >= self.env.unwrapped.max_steps:
                            cause = "max_steps"
                        else:
                            cause = "bad_end"

                    # Update safety tracking and Q-values
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


    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)


    def load(self, path="q_table.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)


    def dump_q_values(self, filename):
        # Write learned Q-values to file for inspection
        action_names = {0: "left", 1: "right", 2: "up", 3: "down"}
        with open(filename, "w") as f:
            for state, q_vals in self.q_table.items():
                _, pos, time_flag, carrying = state
                pos = tuple(int(x) for x in pos)
                time_str = "morning" if time_flag == 0 else "evening"
                carry_str = "carrying" if carrying else "empty"
                f.write(f"State  pos={pos}  time={time_str}  {carry_str}\n")
                for a in range(self.env.action_space.n):
                    name = action_names.get(a, str(a))
                    f.write(f"    {name:>5}:  {q_vals[a]:.4f}\n")
                f.write("\n")


if __name__ == "__main__":
    env = gym.make("BridgeEnv-v0", render_mode=None, phase="train")
    agent = QLearningAgent(env)
    agent.train(episodes=25000)
    agent.save()
