import matplotlib.pyplot as plt
import numpy as np
from CartPoleRL import runCartPole
from CartPoleRLModified import runCartPoleModified
from FrozenLakeRL import runFrozenLake
from FrozenLakeRLModified import runFrozenLakeModified
from HalfCheetaRL import runHalfCheeta
from HalfCheetaRLModified import runHalfCheetaModified
from gymnasium.envs.registration import register

register(
    id='CartPoleModified',
    entry_point='modifiedEnvs.CartPoleEnvModified:CartPoleEnvCustom'
)

register(
    id='FrozenLakeModified',
    entry_point='modifiedEnvs.FrozenLakeModified:FrozenLakeEnvCustom'
)

register(
    id='HalfCheetaModified',
    entry_point='modifiedEnvs.HalfCheetaModified:HalfCheetahEnvCustom'
)

def plot_evaluation_results(names):
    plt.figure(figsize=(10, 5))
    title = ""
    for name in names:
        title += '_'+name
        file_path = f'./Question5/Q5C/data/logs/{name}/evaluations.npz'
        try:
            data = np.load(file_path)

            # Extract the relevant data for plotting
            timesteps = data['timesteps']
            results = data['results']
            mean_rewards = np.mean(results, axis=1)

            # Plotting the results
            plt.plot(timesteps, mean_rewards, marker='o', linestyle='-', label=name)
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping this file.")

    plt.title('Evaluation Results Over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./Question5/Q5C/data/figs/{title}.png')
    plt.close()

runCartPole(300000)
runCartPoleModified(300000)

plot_evaluation_results(['CartPole', 'CartPoleModified'])


runFrozenLake(300000)
runFrozenLakeModified(300000)

plot_evaluation_results(['FrozenLake', 'FrozenLakeModified'])

runHalfCheeta(5000000)
runHalfCheetaModified(5000000)

plot_evaluation_results(['HalfCheeta', 'HalfCheetaModified'])