# -*- coding: utf-8 -*-
from Gym_LineTracer import LineTracerEnv, APPWINDOW
from sample_agent import Agent

import wx
from threading import Thread

# Define Agent And Environment
agent = Agent(2)
env = LineTracerEnv()

class SimulationLoop(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.start()    # start the thread

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
            wx.Yield()

        env.monitor.close()
        
        print("Simulation completed\nWaiting for closing window..")
        
       
if __name__ == '__main__':
    app = wx.PySimpleApp()
    w = APPWINDOW(title='RL Test')
    w.SetWorld(env)
    w.Center()
    w.Show()
    SimulationLoop()
    app.MainLoop()