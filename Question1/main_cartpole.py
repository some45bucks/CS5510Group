import gym
import numpy as np
from gym.envs.registration import register

# register cartpole env for use
register(
    id='CustomCartPole-v1',
    entry_point='cartpole:CartPoleEnv',
)

# returns 1 or 0 for linear direction of cart
def simple_control(Kp, theta, Kd, theta_dot):
    return 1 if Kp * theta + Kd * theta_dot > 0 else 0

# translates reward amount to time
def reward_to_time(reward, tau):
    seconds = reward * tau
    minutes, seconds = divmod(seconds, 60)
    return int(minutes), int(seconds)

# runs the simulation given an environment, control values: Kp and Kd, rendering boolean, and the total number of episodes.
def run_simulation(env, Kp, Kd, render=False, num_episodes=10):
    total_rewards = []

    for episode in range(num_episodes):
        # preparation for a simulation
        observation = env.reset()
        total_reward = 0
        done = False

        # runs the simulation for one episode
        while not done:
            x, x_dot, theta, theta_dot = observation
            action = simple_control(Kp, theta, Kd, theta_dot)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()

            # stopping condition so that an episode does not take too long if it's not rendering.
            if total_reward > 100000 and not render:
                done = True

        total_rewards.append(total_reward)

    env.close()
    return total_rewards  # Return a list of all rewards in the episode.

def find_best(env, Kp_range, Kd_range):
    best_reward = -float("inf")
    best_Kp = None
    best_Kd = None

    for Kp in Kp_range:
        for Kd in Kd_range:
            rewards = run_simulation(env, Kp, Kd, render=False, num_episodes=2000)
            # Calculate the trimmed mean (e.g., trim 10% from each end).
            trim_percent = 0.10 
            num_to_trim = int(len(rewards) * trim_percent)
            trimmed_rewards = np.sort(rewards)[num_to_trim:-num_to_trim]
            trimmed_mean_reward = np.mean(trimmed_rewards)
            
            print(f"Kp: {Kp}, Kd: {Kd}, Trimmed Mean Reward: {trimmed_mean_reward}")

            if trimmed_mean_reward > best_reward:
                best_reward = trimmed_mean_reward
                best_Kp = Kp
                best_Kd = Kd

    return best_Kp, best_Kd, best_reward

env = gym.make("CustomCartPole-v1")

# Kp_range = np.linspace(4.0, 12.0, 5)
# Kd_range = np.linspace(0.2, 1.0, 5)
# best_Kp, best_Kd, best_reward = find_best(env, Kp_range, Kd_range)
# print(f"Best Kp: {best_Kp}, Best Kd: {best_Kd}, Best Average Reward: {best_reward}")

best_Kp = 11.0
best_Kd = 0.3
print("Visual simulation using the best performing values..")
rewards = run_simulation(env, best_Kp, best_Kd, render=True, num_episodes=1)
minutes, seconds = reward_to_time(rewards[0], env.tau)
print(f"Reward: {rewards[0]} Total Time: {minutes} minutes {seconds} seconds")

# print(f"Evaluating {1000} simulations using the best performing values..")
# rewards = run_simulation(env, best_Kp, best_Kd, render=False, num_episodes=1000)
# minutes, seconds = reward_to_time(np.mean(rewards), env.tau)
# print(f"Average Reward: {np.mean(rewards)} Average Time: {minutes} minutes {seconds} seconds")