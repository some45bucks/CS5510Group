import math
import numpy as np
import random

class CircleGym2A:
    inputs = 5
    outputs = 50

    map = [-(i)/50 for i in range(51)]

    def __init__(self,_L,_t,_T,_R,maxV = 1):
        self.L = _L
        self.t = _t
        self.x = 0
        self.y = 0
        self.theta = 0
        self.Time = _T
        self.T=_T
        self.R = _R
        self.Mv = maxV
        self.Ma = math.pi/3

        self.sweepTheta = 0
        self.sweepX = 0
        self.sweepY = self.R

    def generate_random_point_in_circle(self,radius):
        # Generate random angle between 0 and 2pi
        theta = 2 * math.pi * random.random()

        # Generate random radius in the circle using square root of uniform distribution
        # This is done to avoid clustering at the center
        r = radius * math.sqrt(random.random())

        # Convert polar to Cartesian coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        return (x, y)

    def reset(self):
        x,y = self.generate_random_point_in_circle(self.R)
        self.x = 0
        self.y = 0

        self.sweepTheta = 0
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)

        self.theta = 0
        self.T=self.Time

        return np.array([self.x/self.R,self.y/self.R,self.theta/self.Ma,self.sweepX/self.R,self.sweepY/self.R])
    
    def step(self,actions):
        v = 8
        action = self.map[actions]
        a = action* self.Ma

        
        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t
        self.theta = self.theta + (v/self.L)*math.tan(a)*self.t

        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        self.sweepTheta = self.sweepTheta + math.pi/(14)

        self.T -= self.t

        done = self.T <= 0

        

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])
        dist0 = math.dist([0,0],[self.x,self.y])

        # scientifically proven reward function
        reward = .5-dist +100*(1-(self.T/self.Time))

        if dist0 > self.R:
            reward -= 100*(1-(self.T/self.Time))
            done = True

        if dist > 17.5*(self.T/self.Time)+.5:
            done = True

        return np.array([self.x/self.R,self.y/self.R,self.theta/self.Ma,self.sweepX/self.R,self.sweepY/self.R]), reward, done
    
class CircleGym2B:
    inputs = 5
    outputs = 50

    map = [-(i)/50 for i in range(51)]

    def __init__(self,_L,_t,_T,_R,maxV = 1):
        self.L = _L
        self.t = _t
        self.x = 0
        self.y = 0
        self.theta = 0
        self.Time = _T
        self.T=_T
        self.R = _R
        self.Mv = maxV
        self.Ma = math.pi/3

        self.sweepTheta = 0
        self.sweepX = 0
        self.sweepY = self.R

    def generate_random_point_in_circle(self,radius):
        # Generate random angle between 0 and 2pi
        theta = 2 * math.pi * random.random()

        # Generate random radius in the circle using square root of uniform distribution
        # This is done to avoid clustering at the center
        r = radius * math.sqrt(random.random())

        # Convert polar to Cartesian coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        return (x, y)

    def reset(self):
        x,y = self.generate_random_point_in_circle(self.R)
        self.x = 0
        self.y = 0

        self.sweepTheta = 0
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)

        self.theta = 0
        self.T=self.Time

        return np.array([self.x/self.R,self.y/self.R,self.theta/self.Ma,self.sweepX/self.R,self.sweepY/self.R])
    
    def step(self,actions):
        v = 8
        action = self.map[actions]
        a = action* self.Ma

        
        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t
        self.theta = self.theta + (v/self.L)*math.tan(a)*self.t

        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        self.sweepTheta = self.sweepTheta + math.pi/(14)

        self.T -= self.t

        done = self.T <= 0

        

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])
        dist0 = math.dist([0,0],[self.x,self.y])

        # scientifically proven reward function
        reward = .5-dist +100*(1-(self.T/self.Time))

        if dist0 > self.R:
            reward -= 100*(1-(self.T/self.Time))
            done = True

        if dist > 17.5*(self.T/self.Time)+.5:
            done = True

        return np.array([self.x/self.R,self.y/self.R,self.theta/self.Ma,self.sweepX/self.R,self.sweepY/self.R]), reward, done