# -*- coding: utf-8 -*-
from Gym_LineTracer import LineTracerEnv, APPWINDOW
from sample_agent import Agent

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

import sys

# Define Agent And Environment
agent = Agent(2)
env = LineTracerEnv()

class SimulationLoop(QThread):
    def __init__(self):
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        observation = env.reset()
        for t in range(1000):
            env.render()
            print(observation)
            action = agent.act()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        print("Simulation completed\nWaiting for closing window..")
        
       
if __name__ == '__main__':
    app = 0
    app = QApplication(sys.argv)
    
    w = APPWINDOW(SimulationLoop() , title='RL Test')
    w.SetWorld(env)
    w.show()

    sys.exit(app.exec_())
