# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self):
        return np.tanh(np.random.randn(self.dim_action)) # random action
