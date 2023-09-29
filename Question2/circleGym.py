import math

class CircleGym:
    inputs = 2
    outputs = 2

    def __init__(self,_L,_t,_T,_R):
        self.L = _L
        self.t = _t
        self.x = 0
        self.y = 0
        self.theta = 0
        self.Time = _T
        self.T=_T
        self.R = _R
    
    def reset(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.T=self.Time

        return (self.x,self.y)
    
    def step(self,actions):
        v = actions[0]
        a = actions[1]

        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t
        self.theta = self.theta + (v/self.L)*math.cos(a)*self.t

        self.T -= self.t

        dist = math.dist([0,0],[self.x,self.y])

        if dist > self.R:
            reward = -1
        else:
            reward = dist/self.R
            

        return (self.x,self.y), reward, self.T <= 0