import random
from collections import deque
import numpy as np

from CircleGym import CircleGym2A,CircleGym2B
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

SAVE_PATH = "./CarRLRecent.pt"
SAVE_PATH_BEST = "./CarRLBest.pt"

class QualityNN(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(QualityNN, self).__init__()
        #this is setting up the layers with inputs and outputs
        self.layer1 = torch.nn.Linear(observation_space, 428)
        self.layer2 = torch.nn.Linear(428, 428)
        self.layer3 = torch.nn.Linear(428, action_space)

    #feed forward with inputs
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x)

        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)

        x = self.layer3(x)

        # output is the amount of reward expected with this action
        return x

class Memory(object):
    def __init__(self,l):
        self.memory = deque(maxlen=l)
        self.rewards = []
        self.run = []

    def push(self, element, R):
        self.run.append(element)
        self.rewards.append(R)

    def finishRun(self,discount):
        for i in range(len(self.rewards)-2,-1,-1):
            self.rewards[i] = self.rewards[i]+discount*self.rewards[i+1]
        self.memory.append((self.run,self.rewards))
        self.clear()

    def get_batch(self):
        x = random.choice(self.memory)
        return x[0], x[1]

    def clear(self):
        self.run = []
        self.rewards = []

    def __repr__(self):
        return f"Current elements in memory: {len(self.memory)}"

    def __len__(self):
        return len(self.memory)

class Agent(object):
    def __init__(self, environment):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QualityNN(environment.inputs, environment.outputs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=7e-4)

        #decay the randomness over time
        self.decay = 0.999
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

    def update(self, memory_batch,rewards):
        # unpack our batch and convert to tensors
        states, next_states, actions = self.unpack_batch(memory_batch)
        rewards = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        # compute what the output is (old expected qualities)
        old_targets = self.old_targets(states, actions)

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

        return states, next_states, actions

    #helper function
    def update_randomness(self):
        self.randomness *= self.decay
        self.randomness = max(self.randomness, self.min_randomness)

class TestAgent(object):
    def __init__(self, environment):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QualityNN(environment.inputs, environment.outputs).to(self.device)

    def act(self, state):
        # move the state to a Torch Tensor
        state = torch.from_numpy(state).float().to(self.device)

        # find the quality of both actions (expected reward)
        qualities = self.model(state).cpu()

        action = torch.argmax(qualities).item() # just take the action with most expected reward

        # return that action
        return action

def train2A(max_iteration = 3500,logging_iteration = 50):

    learning = []
    best = float("-inf")

    environment = CircleGym2A(10.668,.5,28,18)
    agent = Agent(environment)
    memory = Memory(10000)

    for iteration in range(1, max_iteration + 1):
        done = False
        state = environment.reset()
        reward_log = 0
        #main loop where the agent interacts with the environment
        while not done:
            action = agent.act(state)
            next_state, reward, done, *_ = environment.step(action)

            memory.push((state, next_state, action), reward)
            reward_log += reward
            state = next_state

        if reward_log > best:
            torch.save(agent.model.state_dict(), SAVE_PATH_BEST)
            test2A(False,True)
            best = reward_log
            
            
        memory.finishRun(.95)
        memory_batch = memory.get_batch()
        for i in range(min(500,len(memory_batch))):
            agent.update(memory_batch[0],memory_batch[1])
        agent.update_randomness()

        learning.append(reward_log)
        if iteration % logging_iteration == 0:
            print(f"Iteration: {iteration}")
            print(f"  Best Reward {best}")
            print(f"  Moving-Average reward: {np.mean(learning[-logging_iteration:]):.4f}")
            print(f"  Memory-Buffer Size: {len(memory.memory)}")
            print(f"  Agent Randomness: {agent.randomness:.3f}")
            
            if best > np.mean(learning[-logging_iteration]):
                a = Agent(environment)
                a.model = QualityNN(environment.inputs, environment.outputs)
                a.model.load_state_dict(torch.load(SAVE_PATH_BEST))
                a.model.eval()
                a.model.to(a.device)
                for i in range(100):
                    done = False
                    state = environment.reset()
                    while not done:
                        action = a.act(state)
                        next_state, reward, done, *_ = environment.step(action)

                        memory.push((state, next_state, action),reward)
                        state = next_state

                    memory.finishRun(.95)

            torch.save(agent.model.state_dict(), SAVE_PATH)
            test2A(False,False)
    torch.save(agent.model.state_dict(), SAVE_PATH)
          
    x = np.arange(0, len(learning), logging_iteration)
    y = np.add.reduceat(learning, x) / logging_iteration

    #very very bad but its ok (don't worry if your not using cuda)
    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    sns.lineplot(x=x, y=y)
    plt.title("Car Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

def test2A(show = True,best = False,dots = True):
    environment = CircleGym2A(10.668,.5,28,18)
    agent = TestAgent(environment)
    agent.model = QualityNN(environment.inputs, environment.outputs)
    if best:
        agent.model.load_state_dict(torch.load(SAVE_PATH_BEST))
    else:
        agent.model.load_state_dict(torch.load(SAVE_PATH))
    agent.model.eval()
    agent.model.to(agent.device)
    pointsX = []
    pointsY = []
    thetas = []
    actions = []

    sweepx = []
    sweepy = []

    for iteration in range(1, 2):
        done = False
        state = environment.reset()
        reward_log = 0
        pointsX.append(0)
        pointsY.append(0)
        thetas.append(0)
        sweepx.append(environment.sweepX/environment.R)
        sweepy.append(environment.sweepY/environment.R)
        #main loop where the agent interacts with the environment
        while not done:
            action = agent.act(state)
            actions.append(environment.map[action]*environment.Ma)
            next_state, reward, done, *_ = environment.step(action)
            reward_log += reward
            state = next_state
            pointsX.append(state[0])
            pointsY.append(state[1])
            thetas.append(state[2])
            
            sweepx.append(environment.sweepX/environment.R)
            sweepy.append(environment.sweepY/environment.R)
        actions.append(0)


    #very very bad but its ok (don't worry if your not using cuda)
    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='black', lw=4),
                Line2D([0], [0], color='green', lw=4)]

    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 18, color='r'))
    ax.plot(np.multiply(pointsX,18), np.multiply(pointsY,18))
    if dots:
        ax.scatter(np.multiply(pointsX,18), np.multiply(pointsY,18), c=c, ec='k')
        ax.scatter(np.multiply(sweepx,18), np.multiply(sweepy,18), c=c, ec='k')
    else:
        ax.quiver(np.multiply(pointsX,18), np.multiply(pointsY,18),-np.sin(thetas),np.cos(thetas))
        ax.quiver(np.multiply(pointsX,18), np.multiply(pointsY,18),-np.sin(np.add(actions,thetas)),np.cos(np.add(actions,thetas)), color='g')
    plt.axis([-1.5*18, 1.5*18, -1.5*18, 1.5*18])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    plt.title("Path")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.legend(custom_lines, ['Path', 'Current Trajectory', 'Turn Action'])
    if best:
        fig.savefig("bestPathGraph.png")
    else:
        fig.savefig("pathGraph.png")
    if show:
        plt.show()
    plt.close()

    return pointsX, pointsY, thetas, actions

def train2B(max_iteration = 3500,logging_iteration = 50):

    learning = []
    best = float("-inf")

    environment = CircleGym2B(10.668,.5,28,18)
    agent = Agent(environment)
    memory = Memory(10000)

    for iteration in range(1, max_iteration + 1):
        done = False
        state = environment.reset()
        reward_log = 0
        #main loop where the agent interacts with the environment
        while not done:
            action = agent.act(state)
            next_state, reward, done, *_ = environment.step(action)

            memory.push((state, next_state, action), reward)
            reward_log += reward
            state = next_state

        if reward_log > best:
            torch.save(agent.model.state_dict(), SAVE_PATH_BEST)
            test2B(False,True)
            best = reward_log
            
            
        memory.finishRun(.95)
        memory_batch = memory.get_batch()
        for i in range(min(500,len(memory_batch))):
            agent.update(memory_batch[0],memory_batch[1])
        agent.update_randomness()

        learning.append(reward_log)
        if iteration % logging_iteration == 0:
            print(f"Iteration: {iteration}")
            print(f"  Best Reward {best}")
            print(f"  Moving-Average reward: {np.mean(learning[-logging_iteration:]):.4f}")
            print(f"  Memory-Buffer Size: {len(memory.memory)}")
            print(f"  Agent Randomness: {agent.randomness:.3f}")
            
            if best > np.mean(learning[-logging_iteration]):
                a = Agent(environment)
                a.model = QualityNN(environment.inputs, environment.outputs)
                a.model.load_state_dict(torch.load(SAVE_PATH_BEST))
                a.model.eval()
                a.model.to(a.device)
                for i in range(100):
                    done = False
                    state = environment.reset()
                    while not done:
                        action = a.act(state)
                        next_state, reward, done, *_ = environment.step(action)

                        memory.push((state, next_state, action),reward)
                        state = next_state

                    memory.finishRun(.95)

            torch.save(agent.model.state_dict(), SAVE_PATH)
            test2B(False,False)
    torch.save(agent.model.state_dict(), SAVE_PATH)
          
    x = np.arange(0, len(learning), logging_iteration)
    y = np.add.reduceat(learning, x) / logging_iteration

    #very very bad but its ok (don't worry if your not using cuda)
    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    sns.lineplot(x=x, y=y)
    plt.title("Car Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

def test2B(show = True,best = False,dots = True):
    environment = CircleGym2B(10.668,.5,28,18)
    agent = TestAgent(environment)
    agent.model = QualityNN(environment.inputs, environment.outputs)
    if best:
        agent.model.load_state_dict(torch.load(SAVE_PATH_BEST))
    else:
        agent.model.load_state_dict(torch.load(SAVE_PATH))
    agent.model.eval()
    agent.model.to(agent.device)
    pointsX = []
    pointsY = []
    thetas = []
    actions = []

    sweepx = []
    sweepy = []

    for iteration in range(1, 2):
        done = False
        state = environment.reset()
        reward_log = 0
        pointsX.append(0)
        pointsY.append(0)
        thetas.append(0)
        sweepx.append(environment.sweepX/environment.R)
        sweepy.append(environment.sweepY/environment.R)
        #main loop where the agent interacts with the environment
        while not done:
            action = agent.act(state)
            actions.append(environment.map[action]*environment.Ma)
            next_state, reward, done, *_ = environment.step(action)
            reward_log += reward
            state = next_state
            pointsX.append(state[0])
            pointsY.append(state[1])
            thetas.append(state[2])
            
            sweepx.append(environment.sweepX/environment.R)
            sweepy.append(environment.sweepY/environment.R)
        actions.append(0)


    #very very bad but its ok (don't worry if your not using cuda)
    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='black', lw=4),
                Line2D([0], [0], color='green', lw=4)]

    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 18, color='r'))
    ax.plot(np.multiply(pointsX,18), np.multiply(pointsY,18))
    if dots:
        ax.scatter(np.multiply(pointsX,18), np.multiply(pointsY,18), c=c, ec='k')
        ax.scatter(np.multiply(sweepx,18), np.multiply(sweepy,18), c=c, ec='k')
    else:
        ax.quiver(np.multiply(pointsX,18), np.multiply(pointsY,18),-np.sin(thetas),np.cos(thetas))
        ax.quiver(np.multiply(pointsX,18), np.multiply(pointsY,18),-np.sin(np.add(actions,thetas)),np.cos(np.add(actions,thetas)), color='g')
    plt.axis([-1.5*18, 1.5*18, -1.5*18, 1.5*18])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    plt.title("Path")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.legend(custom_lines, ['Path', 'Current Trajectory', 'Turn Action'])
    if best:
        fig.savefig("bestPathGraph.png")
    else:
        fig.savefig("pathGraph.png")
    if show:
        plt.show()
    plt.close()

    return pointsX, pointsY, thetas, actions