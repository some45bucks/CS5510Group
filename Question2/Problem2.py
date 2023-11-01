import numpy as np
import math

from CircleGym import CircleGym2ASearch,CircleGym2BSearch,CircleGym2CSearch,CircleGym2DSearch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def solve2A(depth=1000):

    environment = CircleGym2ASearch(10.668,.5,28,18)
    environment.reset()

    done = False

    pointsX = []
    pointsY = []
    thetas = []
    actions = []
    ex = []
    ey = []
    error = []
    sweepx = []
    sweepy = []

    while not done:
        high = 1
        low = -1
        mid = 0
        print(environment.T)
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
        ex.append(environment.errorX)
        ey.append(environment.errorY)
        error.append(math.dist([pointsX[-1],pointsY[-1]],[environment.errorX,environment.errorY]))

        _,done = environment.step(bestAction)

        sweepx.append(environment.sweepX)
        sweepy.append(environment.sweepY)

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 18, color='r'))
    ax.plot(pointsX, pointsY)
    ax.plot(ex, ey)
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

    
    fig, ax = plt.subplots()
    plt.title("Error Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error")
    ax.plot(c,error)
    fig.savefig("./Question2/Q2A/Q2A_Error.png")
    plt.close()

    return pointsX, pointsY, thetas, actions, error

def solve2B(depth=1000):

    environment = CircleGym2BSearch(3.048,.5,28,18,8)
    environment.reset()

    done = False

    pointsX = []
    pointsY = []
    thetas = []
    actionsL = []
    actionsR = []
    ex = []
    ey = []
    error = []
    sweepx = []
    sweepy = []

    while not done:
        highL = 1
        lowL = -1
        midL = 0
        highR = 1
        lowR = -1
        midR = 0
        print(environment.T)
        for i in range(depth):

            highActionL = (highL+midL)/2
            lowActionL = (lowL+midL)/2

            highActionR = (highR+midR)/2
            lowActionR = (lowR+midR)/2

            hhReward, _ = environment.step(highActionL,highActionR)
            environment.stepBack()
            hlReward, _ = environment.step(highActionL,lowActionR)
            environment.stepBack()
            lhReward, _ = environment.step(lowActionL,highActionR)
            environment.stepBack()
            llReward, _ = environment.step(lowActionL,lowActionR)
            environment.stepBack()

            maxReward = max(max(hhReward,hlReward),max(lhReward,llReward))

            if hhReward == maxReward:
                lowL = midL
                midL = (highL + lowL)/2

                lowR = midR
                midR = (highR + lowR)/2

                bestActionL = highActionL
                bestActionR = highActionR

            elif hlReward == maxReward:
                lowL = midL
                midL = (highL + lowL)/2

                highR = midR
                midR = (highR + lowR)/2

                bestActionL = highActionL
                bestActionR = lowActionR

            elif lhReward == maxReward:
                highL = midL
                midL = (highL + lowL)/2

                lowR = midR
                midR = (highR + lowR)/2

                bestActionL = lowActionL
                bestActionR = highActionR
            else:
                highL = midL
                midL = (highL + lowL)/2

                highR = midR
                midR = (highR + lowR)/2

                bestActionL = lowActionL
                bestActionR = lowActionR

        pointsX.append(environment.x)
        pointsY.append(environment.y)
        thetas.append(environment.theta)
        actionsL.append(bestActionL)
        actionsR.append(bestActionR)
        ex.append(environment.errorX)
        ey.append(environment.errorY)
        error.append(math.dist([pointsX[-1],pointsY[-1]],[environment.errorX,environment.errorY]))

        _,done = environment.step(bestActionL,bestActionR)

        sweepx.append(environment.sweepX)
        sweepy.append(environment.sweepY)

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 18, color='r'))
    ax.plot(pointsX, pointsY)
    ax.plot(ex, ey)
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

    
    fig, ax = plt.subplots()
    plt.title("Error Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error")
    ax.plot(c,error)
    fig.savefig("./Question2/Q2B/Q2B_Error.png")
    plt.close()

    return pointsX, pointsY, thetas, actionsL, actionsR, error

def solve2C(depth=1000,tname="t1"):
    if tname == "t1":
        t = 1
    elif tname == "t_1":
        t = .1
    elif tname == "t_01":
        t = .01

    environment = CircleGym2CSearch(3.048,t,28,9,4)
    environment.reset()

    done = False

    pointsX = []
    pointsY = []
    thetas = []
    actionsL = []
    actionsR = []
    ex = []
    ey = []
    error = []
    sweepx = []
    sweepy = []

    while not done:
        highL = 1
        lowL = -1
        midL = 0
        highR = 1
        lowR = -1
        midR = 0
        print(environment.T)
        for i in range(depth):

            highActionL = (highL+midL)/2
            lowActionL = (lowL+midL)/2

            highActionR = (highR+midR)/2
            lowActionR = (lowR+midR)/2

            hhReward, _ = environment.step(highActionL,highActionR)
            environment.stepBack()
            hlReward, _ = environment.step(highActionL,lowActionR)
            environment.stepBack()
            lhReward, _ = environment.step(lowActionL,highActionR)
            environment.stepBack()
            llReward, _ = environment.step(lowActionL,lowActionR)
            environment.stepBack()

            maxReward = max(max(hhReward,hlReward),max(lhReward,llReward))

            if hhReward == maxReward:
                lowL = midL
                midL = (highL + lowL)/2

                lowR = midR
                midR = (highR + lowR)/2

                bestActionL = highActionL
                bestActionR = highActionR

            elif hlReward == maxReward:
                lowL = midL
                midL = (highL + lowL)/2

                highR = midR
                midR = (highR + lowR)/2

                bestActionL = highActionL
                bestActionR = lowActionR

            elif lhReward == maxReward:
                highL = midL
                midL = (highL + lowL)/2

                lowR = midR
                midR = (highR + lowR)/2

                bestActionL = lowActionL
                bestActionR = highActionR
            else:
                highL = midL
                midL = (highL + lowL)/2

                highR = midR
                midR = (highR + lowR)/2

                bestActionL = lowActionL
                bestActionR = lowActionR

        pointsX.append(environment.x)
        pointsY.append(environment.y)
        thetas.append(environment.theta)
        actionsL.append(bestActionL)
        actionsR.append(bestActionR)
        ex.append(environment.errorX)
        ey.append(environment.errorY)
        error.append(math.dist([pointsX[-1],pointsY[-1]],[environment.errorX,environment.errorY]))

        _,done = environment.step(bestActionL,bestActionR)

        sweepx.append(environment.sweepX)
        sweepy.append(environment.sweepY)

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 9, color='r'))
    ax.plot(pointsX, pointsY)
    ax.plot(ex, ey)
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

    return pointsX, pointsY, thetas, actionsL, actionsR, error

def solve2D(depth=1000,tname="t1"):
    if tname == "t1":
        t = 1
    elif tname == "t_1":
        t = .1
    elif tname == "t_01":
        t = .01

    environment = CircleGym2DSearch(3.048,t,28,9,4)
    environment.reset()

    done = False

    pointsX = []
    pointsY = []
    thetas = []
    actionsL = []
    actionsR = []
    ex = []
    ey = []
    error = []
    sweepx = []
    sweepy = []

    while not done:
        highL = 1
        lowL = -1
        midL = 0
        highR = 1
        lowR = -1
        midR = 0
        print(environment.T)
        for i in range(depth):

            highActionL = (highL+midL)/2
            lowActionL = (lowL+midL)/2

            highActionR = (highR+midR)/2
            lowActionR = (lowR+midR)/2

            hhReward, _ = environment.step(highActionL,highActionR)
            environment.stepBack()
            hlReward, _ = environment.step(highActionL,lowActionR)
            environment.stepBack()
            lhReward, _ = environment.step(lowActionL,highActionR)
            environment.stepBack()
            llReward, _ = environment.step(lowActionL,lowActionR)
            environment.stepBack()

            maxReward = max(max(hhReward,hlReward),max(lhReward,llReward))

            if hhReward == maxReward:
                lowL = midL
                midL = (highL + lowL)/2

                lowR = midR
                midR = (highR + lowR)/2

                bestActionL = highActionL
                bestActionR = highActionR

            elif hlReward == maxReward:
                lowL = midL
                midL = (highL + lowL)/2

                highR = midR
                midR = (highR + lowR)/2

                bestActionL = highActionL
                bestActionR = lowActionR

            elif lhReward == maxReward:
                highL = midL
                midL = (highL + lowL)/2

                lowR = midR
                midR = (highR + lowR)/2

                bestActionL = lowActionL
                bestActionR = highActionR
            else:
                highL = midL
                midL = (highL + lowL)/2

                highR = midR
                midR = (highR + lowR)/2

                bestActionL = lowActionL
                bestActionR = lowActionR

        pointsX.append(environment.x)
        pointsY.append(environment.y)
        thetas.append(environment.theta)
        actionsL.append(bestActionL)
        actionsR.append(bestActionR)
        ex.append(environment.errorX)
        ey.append(environment.errorY)
        error.append(math.dist([pointsX[-1],pointsY[-1]],[environment.errorX,environment.errorY]))

        _,done = environment.step(bestActionL,bestActionR)

        sweepx.append(environment.sweepX)
        sweepy.append(environment.sweepY)

    import os    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    c = [.5*i for i in range(len(pointsX))]
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), 9, color='r'))
    ax.plot(pointsX, pointsY)
    ax.plot(ex, ey)
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

    return pointsX, pointsY, thetas, actionsL, actionsR, error