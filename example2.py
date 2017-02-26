# -*- coding: utf-8 -*-
from Gym_LineTracer import LineTracerEnv, APPWINDOW
from gym import spaces

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

import numpy as np

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

import sys

class SState(object):
    def __init__(self, STATE_NUM, DIM):
        self.STATE_NUM = STATE_NUM
        self.DIM = DIM
        self.seq = np.zeros((STATE_NUM, DIM), dtype=np.float32)

    def push_s(self, state):
        self.seq[1:self.STATE_NUM ] = self.seq[0:self.STATE_NUM -1]
        self.seq[0] = state

    def reset(self):
        self.seq = np.ones_like(self.seq)

    def fill_s(self, state):
        for i in range(0, self.STATE_NUM):
            self.seq[i] = state

# Define Agent And Environment
class LineTracerEnvDiscrete(LineTracerEnv):
    actions = np.array([[0.5, 0.1], [1.0, 1.0],  [0.1, 0.5]]) 
    OBSDIM = 4

    def __init__(self):
        self.action_space_d = spaces.Discrete(self.actions.shape[0])
        self.observation_space_d = spaces.Discrete(self.OBSDIM)
        self.MyState = SState(self.OBSDIM, 1)

        super().__init__()

    def _step(self, action):
        tempState, tmpReward, tmpDone, tmpInfo = super()._step(self.actions[action])
        self.MyState.push_s(tempState)
        return self.MyState.seq.flatten(), tmpReward, tmpDone, tmpInfo

    def _reset(self):
        s = super()._reset()
        self.MyState.reset()
        return self.MyState.seq.flatten()

env = LineTracerEnvDiscrete()

q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    env.OBSDIM, env.action_space_d.n,
    n_hidden_layers=2, n_hidden_channels=50)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

gamma = 0.95

explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space_d.sample)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_frequency=1,
    target_update_frequency=100)



class SimulationLoop(QThread):
    def __init__(self):
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        n_episodes = 200
        max_episode_len = 5000
        for i in range(1, n_episodes + 1):
            observation = env.reset()
            reward = 0
            R = 0
            for t in range(max_episode_len):
                env.render()
                act = agent.act_and_train(observation, reward)
                observation, reward, done, info = env.step(act)
                R += reward
                if done:
                    print("Episode {} finished after {} timesteps".format(i, t+1))
                    break
            agent.stop_episode_and_train(observation, reward, done)

        agent.save('agent')
        print("Training completed. Now TestMode.")
        for i in range(10):
            observation = env.reset()
            reward = 0
            R = 0
            for t in range(max_episode_len):
                env.render()
                act = agent.act(observation)
                observation, reward, done, info = env.step(act)
                R += reward
                if done:
                    print("Test Episode {} finished after {} timesteps".format(i, t+1))
                    break
            print('test episode:', i, 'R:', R)
            agent.stop_episode()
        print("Test completed. Now Waiting for closing the window.")


if __name__ == '__main__':
    app = 0
    app = QApplication(sys.argv)
    
    w = APPWINDOW(SimulationLoop() , title='RL Test')
    w.SetWorld(env)
    w.show()

    sys.exit(app.exec_())
