import matplotlib.pyplot as plt
import numpy as np


def load_returns(filepath):
    # Load per-episode returns from file
    with open(filepath, "r") as f:
        return [float(line.strip()) for line in f if line.strip()]


def compute_moving_average(data, window_size):
    # Apply moving average for smoothing
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_returns_with_smoothing(returns, window=50):
    # Plot raw and smoothed episode returns to visualize learning curve
    returns_array = np.array(returns)
    moving_avg = compute_moving_average(returns_array, window)

    plt.figure(figsize=(12, 6))
    plt.plot(returns_array, alpha=0.3, label="Episode Return")
    plt.plot(range(window - 1, len(returns_array)), moving_avg, linewidth=2, color="blue", label=f"Smoothed Return (MA-{window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Performance")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_unsafe_deaths(returns, window=50):
    # Identify unsafe episodes (termination by falling into water)
    deaths = [1 if -1099 <= r <= -1000 else 0 for r in returns]
    deaths_array = np.array(deaths)

    # moving_avg = compute_moving_average(deaths_array, window)
    # plt.figure(figsize=(12, 6))
    # plt.plot(deaths_array, alpha=0.3, label="Unsafe Deaths (binary)")
    # plt.plot(range(window - 1, len(deaths_array)), moving_avg, linewidth=2, label=f"Smoothed Unsafe Deaths (MA-{window})")
    # plt.xlabel("Episode")
    # plt.ylabel("Unsafe Death Indicator")
    # plt.title("Unsafe Deaths Over Training")
    # plt.legend(loc="upper right")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Print summary statistics
    total_deaths = np.sum(deaths_array)
    total_episodes = len(returns)
    print(f"Unsafe deaths: {total_deaths} / {total_episodes} ({100.0 * total_deaths / total_episodes:.2f}%)")


if __name__ == "__main__":
    reward_file = "q_learning_rewards.txt"  # Path to reward log
    returns = load_returns(reward_file)

    plot_returns_with_smoothing(returns, window=100)
    #plot_unsafe_deaths(returns, window=100)
