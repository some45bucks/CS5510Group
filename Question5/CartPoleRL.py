import random
from collections import deque
import numpy as np

import gym
import torch

import matplotlib.pyplot as plt
import seaborn as sns

SAVE_PATH = "./saves/CartPoleRecent.pt"


# this is the Neural Network also called a Policy
class QualityNN(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(QualityNN, self).__init__()
        #this is setting up the layers with inputs and outputs
        self.layer1 = torch.nn.Linear(observation_space, 64)
        self.layer2 = torch.nn.Linear(64, 128)
        self.layer3 = torch.nn.Linear(128, action_space)

    #feed forward with inputs
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x)

        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)

        x = self.layer3(x)

        # output is the amount of reward expected with this action
        return x

# we store states and next states in memory adn then train the agent once it's done with a scene
#it acts like a queue so we only train on the x most recent
class Memory(object):
    def __init__(self, max_size=100):
        self.memory = deque(maxlen=max_size)

    def push(self, element):
        self.memory.append(element)

    def get_batch(self, batch_size=4):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        return random.sample(self.memory, batch_size)

    def __repr__(self):
        return f"Current elements in memory: {len(self.memory)}"

    def __len__(self):
        return len(self.memory)

# this is the actual agent that contains the NN
class Agent(object):
    def __init__(self, environment):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = QualityNN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        #decay the randomness over time
        self.decay = 0.9995
        self.randomness = 1.00
        self.min_randomness = 0.001

    def act(self, state):
        # move the state to a Torch Tensor
        state = torch.from_numpy(state).float().to(self.device)

        # find the quality of both actions (expected reward)
        qualities = self.model(state).cpu()

        # sometimes take a random action (so we don't get stuck in local mins as easy)
        if np.random.rand() <= self.randomness:
            action = np.random.randint(low=0, high=qualities.size(dim=0))
        else:
            action = torch.argmax(qualities).item() # just take the action with most expected reward

        # return that action
        return action

    def update(self, memory_batch):
        # unpack our batch and convert to tensors
        states, next_states, actions, rewards = self.unpack_batch(memory_batch)

        # compute what the output is (old expected qualities)
        old_targets = self.old_targets(states, actions)

        # compute what the output should be (new expected qualities)
        # longer version: we save states in pairs in memory and the action that was taken to get from the past state
        # to the future state. we then train the model to predict what wit will predict in the next state if it takes 
        # that action, because the NN should by trying to learn what reward is expected at each action and that should 
        # take into future actions predicted rewards. we also add in the reward it gets just by living another state.
        # doing this the NN should predict the total reward it will get by taking an action, and then choose the action 
        # with the best reward. LMK what you are still confused about on discord.
        new_targets = self.new_targets(states, next_states, rewards, actions)

        # compute the difference between old and new estimates
        loss = torch.nn.MSELoss()
        loss = loss(old_targets, new_targets)

        # apply difference to the neural network through grad descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def old_targets(self, states, actions):
        return self.model(states).gather(1, actions)

    def new_targets(self, states, next_states, rewards, actions):
        return (
            rewards +
            (torch.max(self.model(next_states), dim=1, keepdim=True)[0]) 
        )

    #helper function
    def unpack_batch(self, batch):
        states = [item[0] for item in batch]
        states = torch.tensor(states).float().to(self.device)

        next_states = [item[1] for item in batch]
        next_states = torch.tensor(next_states).float().to(self.device)

        # unsqueeze(1) makes 2d array. [1, 0, 1, ...] -> [[1], [0], [1], ...]
        actions = [item[2] for item in batch]
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)

        rewards = [item[3] for item in batch]
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)

        return states, next_states, actions, rewards

    #helper function
    def update_randomness(self):
        self.randomness *= self.decay
        self.randomness = max(self.randomness, self.min_randomness)

def train(max_iteration = 3500,logging_iteration = 50):

    learning = []

    environment = gym.make("CartPole-v1")
    agent = Agent(environment)
    memory = Memory(max_size=10000)

    for iteration in range(1, max_iteration + 1):
        steps = 0
        done = False
        state = (environment.reset())[0]
        
        #main loop where the agent interacts with the environment
        while not done:
            action = agent.act(state)
            next_state, reward, done, *_ = environment.step(action)

            memory.push(element=(state, next_state, action, reward))

            state = next_state
            steps += 1

        #get some batch size to train on from memory
        memory_batch = memory.get_batch(batch_size=256)
        
        #where the agent trains itself
        agent.update(memory_batch)
        agent.update_randomness()

        learning.append(steps)
        if iteration % logging_iteration == 0:
            print(f"Iteration: {iteration}")
            print(f"  Moving-Average Steps: {np.mean(learning[-logging_iteration:]):.4f}")
            print(f"  Memory-Buffer Size: {len(memory.memory)}")
            print(f"  Agent Randomness: {agent.randomness:.3f}")
            torch.save(agent.model.state_dict(), SAVE_PATH)
    
    torch.save(agent.model.state_dict(), SAVE_PATH)
          
    x = np.arange(0, len(learning), logging_iteration)
    y = np.add.reduceat(learning, x) / logging_iteration

    sns.lineplot(x=x, y=y)
    plt.title("Cart Lifespan During Training")
    plt.xlabel("Episodes")
    plt.ylabel("Lifespan Steps")
    plt.show()

# for testing and generating video
class TestAgent(object):
    def __init__(self, environment):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = QualityNN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)

    def act(self, state):
        # move the state to a Torch Tensor
        state = torch.from_numpy(state).float().to(self.device)

        # find the quality of both actions (expected reward)
        qualities = self.model(state).cpu()

        action = torch.argmax(qualities).item() # just take the action with most expected reward

        # return that action
        return action

def test():
    environment = gym.make("CartPole-v1")
    agent = TestAgent(environment)
    agent.model = QualityNN(environment.observation_space.shape[0], environment.action_space.n)
    agent.model.load_state_dict(torch.load(SAVE_PATH))
    agent.model.eval()

    # will save video for us
    environment = gym.wrappers.RecordVideo(
        gym.make("CartPole-v1", render_mode="rgb_array"),
        video_folder="./videos",
        episode_trigger=lambda x: x == 0
    )

    for iteration in range(1, 2):
        steps = 0
        done = False
        state = (environment.reset())[0]

        while not done:
            steps += 1

            action = agent.act(state)
            state, reward, done, *_ = environment.step(action)