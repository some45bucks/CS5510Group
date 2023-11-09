import math
    
class CircleGym2ASearch:

    def __init__(self,_L,_t,_R):
        self.L = _L
        self.t = _t
        self.R = _R
        self.Ma = math.pi/3

    def reset(self):
        self.x = 0
        self.y = 0
        self.px = 0
        self.py = 0

        self.sweepTheta = 0
        self.sweepX = 0
        self.sweepY = 0

        self.psweepTheta = 0
        self.psweepX = 0
        self.psweepY = 0

        self.theta = -math.pi/2
        self.ptheta = 0

        self.C = math.pi * 2 * self.R
        self.sC = math.pi * (self.R/2)
        self.pC = 0
        self.psC = 0

        self.dist0 = 0
    
    def stepBack(self):
        self.x = self.px
        self.y = self.py

        self.sweepTheta = self.psweepTheta
        self.sweepX = self.psweepX
        self.sweepY = self.psweepY

        self.theta =self.ptheta
        self.C = self.pC
        self.sC = self.psC

        self.dist0 = 0

    def step(self,action):

        self.px = self.x
        self.py = self.y

        self.psweepTheta = self.sweepTheta
        self.psweepX = self.sweepX
        self.psweepY = self.sweepY

        self.ptheta =self.theta
        self.pC=self.C
        self.psC=self.sC

        v = 8
        a = action * self.Ma

        self.theta = self.theta + (v/self.L)*math.tan(a)*self.t
        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t

        if self.sC > 0:
            if self.sC - v * self.t  < 0:
                self.sweepTheta = self.sweepTheta + (self.sC)/(self.R/2)
                self.sweepX = (self.R/2)*math.sin(self.sweepTheta)
                self.sweepY = (self.R/2)*math.cos(self.sweepTheta) - self.R/2

                self.sC = 0

                self.sweepTheta = self.sweepTheta + ((v * self.t - self.sC)/self.R)
                self.sweepX = self.R*math.sin(self.sweepTheta)
                self.sweepY = self.R*math.cos(self.sweepTheta)
                
                self.C -= v * self.t - self.sC
            else:
                self.sweepTheta = self.sweepTheta + ((v*self.t)/(self.R/2))
                self.sweepX = (self.R/2)*math.sin(self.sweepTheta)
                self.sweepY = (self.R/2)*math.cos(self.sweepTheta) - self.R/2
                
                self.sC -= v * self.t
            
        else:

            self.sweepTheta = self.sweepTheta + ((v*self.t)/self.R)
            self.sweepX = self.R*math.sin(self.sweepTheta)
            self.sweepY = self.R*math.cos(self.sweepTheta)
            
            self.C -= v * self.t
            

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        self.dist0 = math.dist([0,0],[self.x,self.y])/self.R

        reward = 1-dist

        if self.dist0 > 1:
            reward = float("-inf")

        done = self.C <= 0

        return reward, done
    
class CircleGym2BSearch:

    def __init__(self,_W,_t,_R):
        self.W = _W
        self.t = _t
        self.R = _R
        self.Mv = 8

    def reset(self):
        self.x = 0
        self.y = 0
        self.px = 0
        self.py = 0

        self.sweepTheta = 0
        self.sweepX = 0
        self.sweepY = 0

        self.psweepTheta = 0
        self.psweepX = 0
        self.psweepY = 0

        self.theta = -math.pi/2
        self.ptheta = 0

        self.C = math.pi * 2 * self.R 
        self.sC = math.pi * ((self.R)/2)
        self.pC = 0
        self.psC = 0

        self.dist0 = 0
    
    def stepBack(self):
        self.x = self.px
        self.y = self.py

        self.sweepTheta = self.psweepTheta
        self.sweepX = self.psweepX
        self.sweepY = self.psweepY

        self.theta =self.ptheta
        self.C = self.pC
        self.sC = self.psC

        self.dist0 = 0

    def step(self,actionL,actionR):

        self.px = self.x
        self.py = self.y

        self.psweepTheta = self.sweepTheta
        self.psweepX = self.sweepX
        self.psweepY = self.sweepY

        self.ptheta =self.theta
        self.pC=self.C
        self.psC=self.sC

        vl = actionL* self.Mv
        vr = actionR* self.Mv

        v = .5 * (vr + vl)

        self.theta = self.theta + (1/self.W)*(vr - vl) * self.t
        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t

        if self.sC > 0:
            if self.sC - abs(v) * self.t  < 0:
                self.sweepTheta = self.sweepTheta + (self.sC)/(self.R/2)
                self.sweepX = (self.R/2)*math.sin(self.sweepTheta)
                self.sweepY = (self.R/2)*math.cos(self.sweepTheta) - self.R/2

                self.sC = 0

                self.sweepTheta = self.sweepTheta + ((abs(v) * self.t - self.sC)/self.R)
                self.sweepX = self.R*math.sin(self.sweepTheta)
                self.sweepY = self.R*math.cos(self.sweepTheta)
                
                self.C -= abs(v) * self.t - self.sC
            else:
                self.sweepTheta = self.sweepTheta + ((abs(v)*self.t)/(self.R/2))
                self.sweepX = (self.R/2)*math.sin(self.sweepTheta)
                self.sweepY = (self.R/2)*math.cos(self.sweepTheta) - self.R/2
                
                self.sC -= abs(v) * self.t
            
        else:
            
            self.sweepTheta = self.sweepTheta + ((abs(v)*self.t)/self.R)
            self.sweepX = self.R*math.sin(self.sweepTheta)
            self.sweepY = self.R*math.cos(self.sweepTheta)
            
            self.C -= abs(v) * self.t
            

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        self.dist0 = math.dist([0,0],[self.x,self.y])/self.R

        reward = (1-dist)

        if self.dist0 > 1:
            reward = -self.dist0

        done = self.C <= 0

        return reward, done
    
class CircleGym2CSearch:

    def __init__(self,_W,_t,_R):
        self.W = _W
        self.t = _t
        self.R = _R
        self.Mv = 8

    def reset(self,angle):
        self.x = 0
        self.y = self.R

        self.sweepTheta = 0
        self.sweepX = 0
        self.sweepY = self.R

        self.theta = -(math.pi/2) * angle

        self.C = math.pi * 2 * self.R 

    def step(self):

        vl = 1* self.Mv
        vr = 0.710376282782* self.Mv

        v = .5 * (vr + vl)

        self.theta = self.theta + (1/self.W)*(vr - vl) * self.t
        self.x = self.x - v*math.sin(self.theta)*self.t
        self.y = self.y + v*math.cos(self.theta)*self.t

        self.sweepTheta = self.sweepTheta + ((abs(v)*self.t)/self.R)
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        
        self.C -= abs(v) * self.t  

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        done = self.C <= 0

        return dist, done
    
class CircleGym2DSearch:

    def __init__(self,_W,_t,_R):
        self.W = _W
        self.t = _t
        self.R = _R
        self.Mv = 8

    def reset(self,angle):
        self.x = 0
        self.y = self.R

        self.sweepTheta = 0
        self.sweepX = 0
        self.sweepY = self.R

        self.theta = -(math.pi/2) * angle

        self.C = math.pi * 2 * self.R 

    def step(self):

        vl = 1* self.Mv
        vr = 0.710376282782* self.Mv

        v = .5 * (vr + vl)

        self.theta = self.theta + (1-.08)*(1/self.W)*(vr - vl) * self.t
        self.x = self.x - (1-.04)*v*math.sin(self.theta)*self.t
        self.y = self.y + (1-.04)*v*math.cos(self.theta)*self.t

        self.sweepTheta = self.sweepTheta + ((abs(v)*self.t)/self.R)
        self.sweepX = self.R*math.sin(self.sweepTheta)
        self.sweepY = self.R*math.cos(self.sweepTheta)
        
        self.C -= abs(v) * self.t  

        dist = math.dist([self.sweepX,self.sweepY],[self.x,self.y])/self.R

        done = self.C <= 0

        return dist, done