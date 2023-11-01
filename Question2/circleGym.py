import math
import numpy as np
import random

class CircleGym2ASearch:

    def __init__(self,_L,_t,_T,_R,maxV = 1):
        self.L = _L
        self.t = _t
        self.Time = _T
        self.T=_T
        self.R = _R
        self.Mv = maxV
        self.Ma = math.pi/3

    def reset(self):
        self.x = 0
        self.y = 0
        self.px = 0
        self.py = 0

        self.sweepTheta = 0
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)

        self.psweepTheta = 0
        self.psweepX = 0
        self.psweepY = 0

        self.errorTheta = 0
        self.errorX = self.x
        self.errorY = self.y

        self.perrorTheta = 0
        self.perrorX = 0
        self.perrorY = 0

        self.theta = 0
        self.ptheta = 0
        self.T=self.Time
        self.pT = 0
    
    def stepBack(self):
        self.x = self.px
        self.y = self.py

        self.sweepTheta = self.psweepTheta
        self.sweepX = self.psweepX
        self.sweepY = self.psweepY

        self.errorTheta = self.perrorTheta
        self.errorX = self.perrorX
        self.errorY = self.perrorY

        self.theta =self.ptheta
        self.T=self.pT

    def step(self,action):

        self.px = self.x
        self.py = self.y

        self.psweepTheta = self.sweepTheta
        self.psweepX = self.sweepX
        self.psweepY = self.sweepY

        self.perrorTheta = self.errorTheta
        self.perrorX = self.errorX
        self.perrorY = self.errorY

        self.ptheta =self.theta
        self.pT=self.T

        v = 8
        a = action * self.Ma

        if math.tan(a) != 0:
            R = self.L/math.tan(a)
            
        else:
            R = 9999999999999

        self.errorTheta = ((v*self.t)/R)
            
        self.errorX = self.errorX + R*math.sin(self.errorTheta + self.theta + math.pi/2) - R*math.sin(self.theta + math.pi/2)
        self.errorY = self.errorY - R*math.cos(self.errorTheta + self.theta + math.pi/2) + R*math.cos(self.theta + math.pi/2)

        self.theta = self.theta + (v/self.L)*math.tan(a)*self.t
        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t

        self.T -= self.t

        done = self.T <= 0

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        self.sweepTheta = self.sweepTheta + ((v*self.t)/self.R)

        dist0 = math.dist([0,0],[self.x,self.y])/self.R

        reward = 1-dist

        if dist0 > 1:
            reward = -1

        return reward, done
    
class CircleGym2BSearch:

    def __init__(self,_W,_t,_T,_R,maxV):
        self.W = _W
        self.t = _t
        self.Time = _T
        self.T=_T
        self.R = _R
        self.Mv = maxV

    def reset(self):
        self.x = 0
        self.y = 0

        self.px = 0
        self.py = 0

        self.sweepTheta = 0
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)

        self.psweepTheta = 0
        self.psweepX = 0
        self.psweepY = 0

        self.errorTheta = 0
        self.errorX = self.x
        self.errorY = self.y

        self.perrorTheta = 0
        self.perrorX = 0
        self.perrorY = 0

        self.theta = 0
        self.ptheta = 0
        self.T=self.Time
        self.pT = 0
    
    def stepBack(self):
        self.x = self.px
        self.y = self.py

        self.sweepTheta = self.psweepTheta
        self.sweepX = self.psweepX
        self.sweepY = self.psweepY

        self.errorTheta = self.perrorTheta
        self.errorX = self.perrorX
        self.errorY = self.perrorY

        self.theta =self.ptheta
        self.T=self.pT

    def step(self,actionL,actionR):

        self.px = self.x
        self.py = self.y

        self.psweepTheta = self.sweepTheta
        self.psweepX = self.sweepX
        self.psweepY = self.sweepY

        self.perrorTheta = self.errorTheta
        self.perrorX = self.errorX
        self.perrorY = self.errorY

        self.ptheta =self.theta
        self.pT=self.T

        vl = actionL* self.Mv
        vr = actionR* self.Mv

        if vr - vl != 0:
            R = (self.W/2)*((vl + vr)/(vr - vl))
            
        else:
            R = 9999999999999

        if R != 0:
            self.errorTheta = (((.5 * (vl + vr))*self.t)/R)

        self.errorX = self.errorX + R*math.sin(self.errorTheta + self.theta + math.pi/2) - R*math.sin(self.theta + math.pi/2)
        self.errorY = self.errorY - R*math.cos(self.errorTheta + self.theta + math.pi/2) + R*math.cos(self.theta + math.pi/2)

        self.theta = self.theta + (1/self.W)*(vr - vl) * self.t
        self.x = self.x - .5 * (vl + vr)*math.sin(self.theta)*self.t
        self.y = self.y + .5 * (vl + vr)*math.cos(self.theta)*self.t

        self.T -= self.t

        done = self.T <= 0

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        self.sweepTheta = self.sweepTheta + ((.70*self.Mv*self.t)/self.R)

        dist0 = math.dist([0,0],[self.x,self.y])/self.R
        
        reward = 1-dist

        if dist0 > 1:
            reward = -1

        return reward, done
    
class CircleGym2CSearch:

    def __init__(self,_W,_t,_T,_R,maxV):
        self.W = _W
        self.t = _t
        self.Time = _T
        self.T=_T
        self.R = _R
        self.Mv = maxV

    def reset(self):
        self.x = 0
        self.y = self.R -.01

        self.px = 0
        self.py = 0

        self.sweepTheta = 0
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)

        self.psweepTheta = 0
        self.psweepX = 0
        self.psweepY = 0

        self.errorTheta = 0
        self.errorX = self.x
        self.errorY = self.y

        self.perrorTheta = 0
        self.perrorX = 0
        self.perrorY = 0

        self.theta = -math.pi/2
        self.ptheta = 0
        self.T=self.Time
        self.pT = 0
    
    def stepBack(self):
        self.x = self.px
        self.y = self.py

        self.sweepTheta = self.psweepTheta
        self.sweepX = self.psweepX
        self.sweepY = self.psweepY

        self.errorTheta = self.perrorTheta
        self.errorX = self.perrorX
        self.errorY = self.perrorY

        self.theta =self.ptheta
        self.T=self.pT

    def step(self,actionL,actionR):

        self.px = self.x
        self.py = self.y

        self.psweepTheta = self.sweepTheta
        self.psweepX = self.sweepX
        self.psweepY = self.sweepY

        self.perrorTheta = self.errorTheta
        self.perrorX = self.errorX
        self.perrorY = self.errorY

        self.ptheta =self.theta
        self.pT=self.T

        vl = actionL* self.Mv
        vr = actionR* self.Mv

        if vr - vl != 0:
            R = (self.W/2)*((vl + vr)/(vr - vl))
            
        else:
            R = 9999999999999

        if R != 0:
            self.errorTheta = (((.5 * (vl + vr))*self.t)/R)

        self.errorX = self.errorX + R*math.sin(self.errorTheta + self.theta + math.pi/2) - R*math.sin(self.theta + math.pi/2)
        self.errorY = self.errorY - R*math.cos(self.errorTheta + self.theta + math.pi/2) + R*math.cos(self.theta + math.pi/2)

        self.theta = self.theta + (1/self.W)*(vr - vl) * self.t
        self.x = self.x - .5 * (vl + vr)*math.sin(self.theta)*self.t
        self.y = self.y + .5 * (vl + vr)*math.cos(self.theta)*self.t

        self.T -= self.t

        done = self.T <= 0

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        
        self.sweepTheta = self.sweepTheta + (.7*self.Mv*self.t/self.R)

        dist0 = math.dist([0,0],[self.x,self.y])/self.R
        
        reward = 1-dist

        if dist0 > 1:
            reward = -1

        return reward, done
    
class CircleGym2DSearch:

    def __init__(self,_W,_t,_T,_R,maxV):
        self.W = _W
        self.t = _t
        self.Time = _T
        self.T=_T
        self.R = _R
        self.Mv = maxV

    def reset(self):
        self.x = 0
        self.y = self.R -.01

        self.px = 0
        self.py = 0

        self.sweepTheta = 0
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)

        self.psweepTheta = 0
        self.psweepX = 0
        self.psweepY = 0

        self.errorTheta = 0
        self.errorX = self.x
        self.errorY = self.y

        self.perrorTheta = 0
        self.perrorX = 0
        self.perrorY = 0

        self.theta = -math.pi/2
        self.ptheta = 0
        self.T=self.Time
        self.pT = 0
    
    def stepBack(self):
        self.x = self.px
        self.y = self.py

        self.sweepTheta = self.psweepTheta
        self.sweepX = self.psweepX
        self.sweepY = self.psweepY

        self.errorTheta = self.perrorTheta
        self.errorX = self.perrorX
        self.errorY = self.perrorY

        self.theta =self.ptheta
        self.T=self.pT

    def step(self,actionL,actionR):

        self.px = self.x
        self.py = self.y

        self.psweepTheta = self.sweepTheta
        self.psweepX = self.sweepX
        self.psweepY = self.sweepY

        self.perrorTheta = self.errorTheta
        self.perrorX = self.errorX
        self.perrorY = self.errorY

        self.ptheta =self.theta
        self.pT=self.T

        vl = actionL* self.Mv
        vr = actionR* self.Mv

        if vr - vl != 0:
            R = (self.W/2)*((vl + vr)/(vr - vl))
            
        else:
            R = 9999999999999

        if R != 0:
            self.errorTheta = ((((1-.04) * .5 * (vl + vr))*self.t)/R)

        self.errorX = self.errorX + (R*math.sin(self.errorTheta + (1-.08)*self.theta + math.pi/2) - R*math.sin((1-.08)*self.theta + math.pi/2))
        self.errorY = self.errorY + (-R*math.cos(self.errorTheta + (1-.08)*self.theta + math.pi/2) + R*math.cos((1-.08)*self.theta + math.pi/2))

        self.theta = self.theta + (1/self.W)*(vr - vl) * self.t
        self.x = self.x - .5 * (vl + vr)*math.sin(self.theta)*self.t
        self.y = self.y + .5 * (vl + vr)*math.cos(self.theta)*self.t

        self.T -= self.t

        done = self.T <= 0

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        
        self.sweepTheta = self.sweepTheta + (.7*self.Mv*self.t/self.R)

        dist0 = math.dist([0,0],[self.x,self.y])/self.R
        
        reward = 1-dist

        if dist0 > 1:
            reward = -1

        return reward, done