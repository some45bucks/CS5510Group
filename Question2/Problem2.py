import numpy as np
import time

from CircleGym import CircleGym2ASearch,CircleGym2BSearch,CircleGym2CSearch,CircleGym2DSearch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def solve2A(depth=1000):

    environment = CircleGym2ASearch(10.668,.5,18)
    environment.reset()

    done = False

    pointsX = []
    pointsY = []
    thetas = []
    actions = []
    dist = []

    while not done:
        high = 1
        low = -1
        mid = 0
        for i in range(depth):

            highAction = (high+mid)/2
            lowAction = (low+mid)/2

            highReward, _ = environment.step(highAction)
            environment.stepBack()
            lowReward, _ = environment.step(lowAction)
            environment.stepBack()

            if highReward > lowReward:
                low = mid
                mid = (high + low)/2
                bestAction = highAction
            else:
                high = mid
                mid = (high + low)/2
                bestAction = lowAction

        pointsX.append(environment.x)
        pointsY.append(environment.y)
        thetas.append(environment.theta)
        actions.append(bestAction)
        

        _,done = environment.step(bestAction)

        dist.append(environment.dist0 * environment.R)
        
        dist.append

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 18, color='r'))
    ax.plot(pointsX, pointsY)
    ax.quiver(pointsX, pointsY,-np.sin(thetas),np.cos(thetas))


    custom_lines = [Line2D([0], [0], color='blue', lw=4),
            Line2D([0], [0], color='orange', lw=4),]
    
    ax.legend(custom_lines, ['Predicted Path', 'True Path'])

    plt.axis([-1.5*18, 1.5*18, -1.5*18, 1.5*18])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    plt.title("Path")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig("./Question2/Q2A/Q2A_Path.png")
    plt.close()

    return pointsX, pointsY, thetas, actions, dist

def solve2B(depth=1000):

    environment = CircleGym2BSearch(3.048,.5,18)
    environment.reset()

    done = False

    pointsX = []
    pointsY = []
    thetas = []
    actionsL = []
    actionsR = []
    dist = []

    while not done:
        high = 1
        low = 0
        mid = (high + low)/2
        for i in range(depth):

            highAction = (high+mid)/2
            lowAction = (low+mid)/2

            hReward, _ = environment.step(1,highAction)
            environment.stepBack()
            lReward, _ = environment.step(1,lowAction)
            environment.stepBack()

            maxReward = max(hReward,lReward)

            if hReward == maxReward:
                low = mid
                mid = (high + low)/2

                bestAction = highAction

            else:
                high = mid
                mid = (high + low)/2

                bestAction = lowAction

        pointsX.append(environment.x)
        pointsY.append(environment.y)
        
        thetas.append(environment.theta)
        actionsL.append(1)
        actionsR.append(bestAction)

        _,done = environment.step(1,bestAction)

        dist.append(environment.dist0 * environment.R)

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 18, color='r'))
    ax.plot(pointsX, pointsY)
    ax.quiver(pointsX, pointsY,-np.sin(thetas),np.cos(thetas))

    custom_lines = [Line2D([0], [0], color='blue', lw=4),
            Line2D([0], [0], color='orange', lw=4),]
    
    ax.legend(custom_lines, ['Predicted Path', 'True Path'])

    plt.axis([-1.5*18, 1.5*18, -1.5*18, 1.5*18])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    plt.title("Path")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig("./Question2/Q2B/Q2B_Path.png")
    plt.close()

    return pointsX, pointsY, thetas, actionsL, actionsR, dist

def solve2C(tname="t1"):
    if tname == "t1":
        t = 1
        angle = 0.764
    elif tname == "t_1":
        t = .1
        angle = .976
    elif tname == "t_01":
        t = .01
        angle = .998

    environment = CircleGym2CSearch(3.048,t,9)

    

    totalError = 0
    t0 = time.time()

    for i in range(100):
        done = False
        environment.reset(angle)

        pointsX = []
        pointsY = []
        pointsXreal = []
        pointsYreal = []
        thetas = []

        count = 0
        error = 0
        while not done:
            pointsX.append(environment.x)
            pointsY.append(environment.y)
            thetas.append(environment.theta)
            e, done = environment.step()
            pointsXreal.append(environment.sweepX)
            pointsYreal.append(environment.sweepY)
            error += e
            count += 1

        totalError += error/count

    t1 = time.time()

    totalTime = t1-t0

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 9, color='r'))
    ax.plot(pointsX, pointsY)
    ax.plot(pointsXreal, pointsYreal)
    ax.quiver(pointsX, pointsY,-np.sin(thetas),np.cos(thetas))

    custom_lines = [Line2D([0], [0], color='blue', lw=4),
            Line2D([0], [0], color='orange', lw=4),]
    
    ax.legend(custom_lines, ['Predicted Path', 'True Path'])

    plt.axis([-1.5*9, 1.5*9, -1.5*9, 1.5*9])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    plt.title("Path")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig(f"./Question2/Q2C/{tname}/Q2C_{tname}_Path.png")
    plt.close()

    return pointsX, pointsY, thetas, pointsXreal, pointsYreal, totalError, totalTime

def solve2D(tname="t1"):
    if tname == "t1":
        t = 1
        angle = 0.764
    elif tname == "t_1":
        t = .1
        angle = .976
    elif tname == "t_01":
        t = .01
        angle = .998

    environment = CircleGym2DSearch(3.048,t,9)

    

    totalError = 0
    t0 = time.time()

    for i in range(100):
        done = False
        environment.reset(angle)

        pointsX = []
        pointsY = []
        pointsXreal = []
        pointsYreal = []
        thetas = []

        count = 0
        error = 0
        while not done:
            pointsX.append(environment.x)
            pointsY.append(environment.y)
            thetas.append(environment.theta)
            e, done = environment.step()
            pointsXreal.append(environment.sweepX)
            pointsYreal.append(environment.sweepY)
            error += e
            count += 1

        totalError += error/count

    t1 = time.time()

    totalTime = t1-t0

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 9, color='r'))
    ax.plot(pointsX, pointsY)
    ax.plot(pointsXreal, pointsYreal)
    ax.quiver(pointsX, pointsY,-np.sin(thetas),np.cos(thetas))

    custom_lines = [Line2D([0], [0], color='blue', lw=4),
            Line2D([0], [0], color='orange', lw=4),]
    
    ax.legend(custom_lines, ['Predicted Path', 'True Path'])

    plt.axis([-1.5*9, 1.5*9, -1.5*9, 1.5*9])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    plt.title("Path")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig(f"./Question2/Q2D/{tname}/Q2D_{tname}_Path.png")
    plt.close()

    return pointsX, pointsY, thetas, pointsXreal, pointsYreal, totalError, totalTime