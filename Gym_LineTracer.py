# -*- coding: utf-8 -*-
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import *

class Walls(object):
    def __init__(self, x0, y0, x1, y1):
        self.xList = [x0, x1]
        self.yList = [y0, y1]
        self.P_color = QColor(50,50,50)

    def addPoint(self, x, y):
        self.xList.append(x)
        self.yList.append(y)

    def Draw(self,dc):
        dc.setPen(self.P_color)
        for i in range(0, len(self.xList)-1):
            dc.drawLine(self.xList[i], self.yList[i], self.xList[i+1],self.yList[i+1])

    def IntersectLine(self, p0, v0, i):
        dp = [p0[0] - self.xList[i], p0[1] - self.yList[i]]
        v1 = [self.xList[i+1] - self.xList[i], self.yList[i+1] - self.yList[i]]

        denom = float(v1[1]*v0[0] - v1[0]*v0[1])
        if denom == 0.0:
            return [False, 1.0]

        ua = (v1[0] * dp[1] - v1[1] * dp[0])/denom
        ub = (v0[0]*dp[1] - v0[1] * dp[0])/denom

        if 0 < ua and ua< 1.0 and 0 < ub and ub < 1.0:
            return [True, ua]

        return [False, 1.0]

    def IntersectLines(self, p0, v0):
        tmpt = 1.0
        tmpf = False
        for i in range(0, len(self.xList)-1):
            f,t = self.IntersectLine( p0, v0, i)
            if f:
                tmpt = min(tmpt, t)
                tmpf = True
 
        return [tmpf, tmpt]
        
    def adLine(self, p0, i):
        dp = [p0[0] - self.xList[i], p0[1] - self.yList[i]]

        v  = [self.xList[i+1] - self.xList[i], self.yList[i+1] - self.yList[i]]
        vl = (v[0]**2+v[1]**2)

        if(vl == 0.0):
            p1l = (dp[0]**2+dp[1]**2)**0.5
        else:
            t = max(0.0, min(1.0, (dp[0]*v[0] + dp[1]*v[1])/vl))

            p1 = [self.xList[i] + t * v[0] - p0[0], self.yList[i] + t * v[1] - p0[1]]
            p1l = (p1[0]**2+p1[1]**2)**0.5

        return p1l

    def adLines(self, p0, d):
        for i in range(0, len(self.xList)-1):
            if self.adLine( p0, i) <= d:
                return True
        return False

class Ball(object):
    def __init__(self, x, y, color, property = 0):
        self.pos_x = x
        self.pos_y = y
        self.rad = 10 
        
        self.property = property

        self.B_color = color
        self.P_color = QColor(50,50,50)

    def Draw(self, dc):
        dc.setPen(self.P_color)
        dc.setBrush(self.B_color)
        dc.drawEllipse(QPoint(self.pos_x, self.pos_y),self.rad, self.rad)
    
    def SetPos(self, x, y):
        self.pos_x = x
        self.pos_y = y
        
    def IntersectBall(self, p0, v0):
        # StackOverflow:Circle line-segment collision detection algorithm?
        # http://goo.gl/dk0yO1

        o = [-self.pos_x + p0[0], -self.pos_y + p0[1]]
                
        a = v0[0] ** 2 + v0[1] **2
        b = 2 * (o[0]*v0[0]+o[1]*v0[1])
        c = o[0] ** 2 + o[1] **2 - self.rad ** 2
        
        discriminant = float(b * b - 4 * a * c)
        
        if discriminant < 0:
            return [False, 1.0]
        
        discriminant = discriminant ** 0.5
        
        t1 = (- b - discriminant)/(2*a)
        t2 = (- b + discriminant)/(2*a)
        
        if t1 >= 0 and t1 <= 1.0:
            return [True, t1]

        if t2 >= 0 and t2 <= 1.0:
            return [True, t2]

        return [False, 1.0] 
        
class Sens(object):
    def __init__(self):
        self.OffSetAngle   = 0.0
        self.OverHang      = 40.0
        self.obj           = -1

class Agent(Ball):
    def __init__(self, canvasSize, x, y, epsilon = 0.99, model = None):
        super(Agent, self).__init__(
            x, y, QColor(112,146,190)
        )
        self.dir_Angle = 0.0#-math.pi/2.0
        self.speed     = 5.0
        
        self.pos_x_max, self.pos_y_max = canvasSize
        self.pos_y_max = 480
       
        self.EYE = Sens()

    def Sens(self, Course):
        p = [self.pos_x + self.EYE.OverHang*math.cos(self.dir_Angle + self.EYE.OffSetAngle),
             self.pos_y - self.EYE.OverHang*math.sin(self.dir_Angle + self.EYE.OffSetAngle)]
         
        # Line Width = 6.0
        if Course.adLines(p, 5.0):
            self.EYE.obj = 1
        else:
            self.EYE.obj = -1
    
    def Draw(self, dc):
        dc.setPen(self.P_color)
        dc.drawLine(self.pos_x, self.pos_y, 
                self.pos_x + self.EYE.OverHang*math.cos(self.dir_Angle + self.EYE.OffSetAngle),
                self.pos_y - self.EYE.OverHang*math.sin(self.dir_Angle + self.EYE.OffSetAngle))
        super(Agent, self).Draw(dc)
       
    def Move(self, WallsList):
        HitBoundary = False
        
        dp = [ self.speed * math.cos(self.dir_Angle),
              -self.speed * math.sin(self.dir_Angle)]
              
        for w in WallsList:
            if w.IntersectLines([self.pos_x, self.pos_y], dp)[0]:
                dp = [0.0, 0.0]
                HitBoundary = True

        self.pos_x += dp[0] 
        self.pos_y += dp[1]

        if not(self.pos_x > 0 and self.pos_x < self.pos_x_max
                and self.pos_y > 0 and self.pos_y < self.pos_y_max):
            HitBoundary = True

        return HitBoundary
       
    def HitBall(self, b):
        if ((b.pos_x - self.pos_x)**2+(b.pos_y - self.pos_y)**2)**0.5 < (self.rad + b.rad):
            return True
        return False


logger = logging.getLogger(__name__)

class LineTracerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Initializing Course : predfined Oval Course
        # ToDo: 外部ファイル読み込み対応
        Rad = 190.0
        Poly = 16
        self.Course = Walls(240, 50, 640-(50+Rad),50)
        for i in range(1, Poly):
            self.Course.addPoint(Rad*math.cos(-np.pi/2.0 + np.pi*i/Poly)+640-(50+Rad), 
                                Rad*math.sin(-np.pi/2.0 + np.pi*i/Poly)+50+Rad)
        self.Course.addPoint(240, 50+Rad*2)
        for i in range(1, Poly):
            self.Course.addPoint(Rad*math.cos(np.pi/2.0 + np.pi*i/Poly)+(50+Rad), 
                                Rad*math.sin(np.pi/2.0 + np.pi*i/Poly)+50+Rad)
        self.Course.addPoint(240,50)
        
        # Outr Boundary Box
        self.BBox = Walls(640, 479, 0, 479)
        self.BBox.addPoint(0,0)
        self.BBox.addPoint(640,0)
        self.BBox.addPoint(640,479)
        
        # Mono Sensor Line Follower 
        self.A = Agent((640, 480), 240, 49)

        # Action Space : left wheel speed, right wheel speed
        # Observation Space : Detect Line (True, False)
        self.action_space = spaces.Box( np.array([-1.,-1.]), np.array([+1.,+1.])) 
        self.observation_space = spaces.Discrete(1)

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # Action Step
        self.A.speed = (action[0]+action[1])/2.0 * 5.0
        self.A.dir_Angle += math.atan((action[0] - action[1]) * self.A.speed / 2.0 / 5.0)
        self.A.dir_Angle = ( self.A.dir_Angle + np.pi) % (2 * np.pi ) - np.pi
        done = self.A.Move([self.BBox])
        self.A.Sens(self.Course)
        self.state = (1,) if self.A.EYE.obj == 1 else (0,)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Robot just went out over the boundary
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {'AgentPos':(self.A.pos_x,self.A.pos_y),'AegntDir':self.A.dir_Angle}

    def _reset(self):
        self.state = (1,)
        self.A.pos_x = 240.0
        self.A.pos_y = 50.0
        self.A.dir_Angle = 0.0
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        return

class APPWINDOW(QWidget):
    def __init__(self, WorkerThread, parent=None, id=-1, title=None):
        super().__init__()
        
        self.resize(640, 480)
        self.setWindowTitle(title)
        self.setStyleSheet("background-color : white;"); 
               
        self.World = None
 
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.OnTimer)
        self.timer.start(20)
        
        self.WThread = WorkerThread
        self.WThread.start()
    
    def SetWorld(self, World):
        self.World = World

    def paintEvent(self, QPaintEvent):
        if self.World is not None:
            # Graphics Update
            qp = QPainter(self)
            
            qp.setPen(QColor(Qt.white))
            qp.setBrush(QColor(Qt.white))
            qp.drawRect(0,0,640,480)
            
            for ag in [self.World.A]:
                ag.Draw(qp)
            self.World.BBox.Draw(qp)
            self.World.Course.Draw(qp)
            
    def OnTimer(self):
        self.update()